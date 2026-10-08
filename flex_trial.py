"""
Flexible single-trial runner for the Round 5 experiments.

One function, run_flex_trial(), generalizes verified_pipeline.run_one_trial,
logit_quantile_infiltration.run_one_trial_logit and
persistent_optimizer_oracle.run_persistent_oracle_trial so every Round 5
variant differs only by keyword arguments:

  space      'prob' (sigmoid output, as in the paper) or 'logit' (pre-sigmoid)
  aqt        use a sliding-buffer quantile threshold (else a static threshold)
  preseed    list of source-domain scores in the SAME space (None/[] = no pre-seed)
  adapt      'none' | 'oracle' | 'oracle_gated' | 'pseudo' | 'pseudo_neg' | 'pseudo_pos'
             (oracle_gated = true labels, but only on the samples the pseudo-label
             gate would select; separates WHICH samples from WHICH labels)
  opt        'reset_adam' (verified_pipeline's per-step re-created Adam),
             'persist_adam', 'persist_sgd'  (constructed once; see
             persistent_optimizer_oracle.py for the group/lr convention)
  lr_mult    multiplier on BASE_LR
  q, window  AQT quantile and buffer size
  static_tau threshold when aqt=False (default 0.5 prob / 0.0 logit)
  floor_tau  optional lower bound on the AQT threshold (hybrid rule)
  cap_tau    optional upper bound on the AQT threshold (hybrid rule)
  inject     optional callable (t, x_t, y_t) -> (x_t, y_t, tag) applied BEFORE
             scoring, used by the adversarial experiments

The loss is BCELoss on the sigmoid output in every case, as in the paper.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from verified_pipeline import (
    CompactMLP, BASE_LR, PHASE1_CUTOFF, PSEUDO_HI_QUANTILE, PSEUDO_LO_QUANTILE,
    sigmoid_weight, AQT_WINDOW, AQT_Q,
)

torch.set_num_threads(1)

CHECKPOINTS = [100, 1000, 5000, 10000, 20000]


def model_logit(model, x):
    return model.out[0](model.block2(model.block1(x)))


def score_buffer(model, X, space, window=AQT_WINDOW, seed=42):
    """Source pre-seed buffer in the requested score space (same sampling as
    verified_pipeline.build_source_pred_buffer)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), window, replace=False)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X[idx], dtype=torch.float32)
        s = model_logit(model, x) if space == "logit" else model(x)
    return s.numpy().flatten().tolist()


def run_flex_trial(state, input_dim, X, y, n, *, space="logit", aqt=True, preseed=None,
                   adapt="none", opt="reset_adam", lr_mult=1.0, q=AQT_Q, window=AQT_WINDOW,
                   static_tau=None, floor_tau=None, cap_tau=None, trace_every=0,
                   label_frac=None, label_seed=0):
    model = CompactMLP(input_dim=input_dim)
    model.load_state_dict(state)
    lr = BASE_LR * lr_mult
    default_tau = 0.0 if space == "logit" else 0.5
    if static_tau is None:
        static_tau = default_tau

    buffer = list(preseed) if (aqt and preseed) else []
    criterion = nn.BCELoss()

    persistent = opt.startswith("persist")
    optimizer = None
    if persistent and adapt != "none":
        cls = torch.optim.Adam if opt == "persist_adam" else torch.optim.SGD
        optimizer = cls([{"params": model.output_params(), "lr": lr},
                         {"params": model.hidden_params(), "lr": 0.0}])

    preds = np.zeros(n, dtype=np.int8)
    ys = np.zeros(n, dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    taus = np.zeros(n, dtype=np.float32)
    n_updates = 0
    lab_stats = {"pos": 0, "pos_correct": 0, "neg": 0, "neg_correct": 0}
    trace = []
    lab_rng = np.random.RandomState(label_seed)  # Round 7: label-budget policies

    for t in range(1, n + 1):
        x_t = torch.tensor(X[t - 1:t], dtype=torch.float32)
        y_t = float(y[t - 1])
        ys[t - 1] = y_t

        model.eval()
        with torch.no_grad():
            s_t = (model_logit(model, x_t) if space == "logit" else model(x_t)).item()
        scores[t - 1] = s_t

        if aqt:
            buffer.append(s_t)
            tau = float(np.quantile(buffer[-window:], q)) if len(buffer) >= window else default_tau
            if floor_tau is not None:
                tau = max(tau, floor_tau)
            if cap_tau is not None:
                tau = min(tau, cap_tau)
        else:
            tau = static_tau
        taus[t - 1] = tau
        preds[t - 1] = 1 if s_t > tau else 0

        if adapt != "none":
            label = None
            if adapt == "oracle":
                label = y_t
                if label_frac is not None and lab_rng.rand() >= label_frac:
                    label = None  # label only a random fraction of flows
            elif adapt == "oracle_benign":
                label = y_t if y_t == 0 else None  # only trusted-benign labels (no attack labels)
            elif adapt == "oracle_alert":
                label = y_t if preds[t - 1] == 1 else None  # analyst triage: label alerted flows only
            else:
                if len(buffer) >= window:
                    recent = buffer[-window:]
                    hi, lo = np.quantile(recent, PSEUDO_HI_QUANTILE), np.quantile(recent, PSEUDO_LO_QUANTILE)
                    if adapt == "oracle_gated":
                        # TRUE label, but only on the samples the pseudo-label gate selects
                        if s_t > hi or s_t < lo:
                            label = y_t
                    elif s_t > hi and adapt in ("pseudo", "pseudo_pos"):
                        label = 1.0
                    elif s_t < lo and adapt in ("pseudo", "pseudo_neg"):
                        label = 0.0
                if label is not None and adapt != "oracle_gated":
                    key = "pos" if label == 1.0 else "neg"
                    lab_stats[key] += 1
                    lab_stats[key + "_correct"] += int(label == y_t)
            if label is not None:
                n_updates += 1
                model.train()
                omega = sigmoid_weight(t)
                if persistent:
                    optimizer.param_groups[0]["lr"] = lr
                    optimizer.param_groups[1]["lr"] = 0.0 if t <= PHASE1_CUTOFF else 0.1 * lr * omega
                else:
                    if t <= PHASE1_CUTOFF:
                        optimizer = torch.optim.Adam(model.output_params(), lr=lr)
                    else:
                        optimizer = torch.optim.Adam([
                            {"params": model.output_params(), "lr": lr},
                            {"params": model.hidden_params(), "lr": 0.1 * lr * omega}])
                optimizer.zero_grad()
                loss = criterion(model(x_t), torch.tensor([[label]], dtype=torch.float32))
                loss.backward()
                optimizer.step()

        if trace_every and t % trace_every == 0:
            trace.append({"t": t, "bias": float(model.out[0].bias.item()),
                          "mean_score_last": float(np.mean(scores[max(0, t - 500):t])),
                          "tau": float(tau)})

    return dict(preds=preds, y=ys, scores=scores, taus=taus, n_updates=n_updates,
                lab_stats=lab_stats, trace=trace)


def metrics_at(res, ns=CHECKPOINTS):
    """macro-F1, attack-F1, recall, precision, alert rate at each checkpoint n
    (computed on the first n samples of the stream)."""
    out = {}
    for n in ns:
        p, y = res["preds"][:n], res["y"][:n]
        tp = float(((p == 1) & (y == 1)).sum()); fp = float(((p == 1) & (y == 0)).sum())
        fn = float(((p == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        af1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        out[str(n)] = {"macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
                       "attack_f1": af1, "recall": rec, "precision": prec,
                       "alert_rate": float(p.mean())}
    return out
