"""Round 8 R3-3: how much of the surrogate decoy attack depends on sharing the victim's source data?
Wednesday pair (raw77). The victim is pretrained on half A of the source. Surrogates (a different seed each):
  same_A      trained on the same half A (full knowledge of the training data)
  disjoint_B  trained on the other half B (no shared flows)
  small_B10   trained on 10% of B
  random      (control) uniformly random benign decoys
Decoys = benign flows in the top 1% of benign scores under the surrogate's scores, inserted at rates
0, 1, 5% of the stream; recall on real attacks (1% of the stream) for AQT, plus the share of decoys that
land in the victim's own top 1% of benign scores. 3 seeds x 8 streams. -> round8_surrogate.json"""
import json
import numpy as np, torch
from verified_pipeline import pretrain_source_model
from flex_trial import score_buffer
from grid_runner import load_pool
from threshold_baselines import scores_of, aqt_preds
from adversarial_preseeded import real_attack_metrics
torch.set_num_threads(1)
N = 20000
X17, y17, X18, y18 = load_pool("raw77")
dim = X17.shape[1]; out = {}
for seed in (42, 123, 456):
    rs = np.random.RandomState(seed); perm = rs.permutation(len(X17)); A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
    B10 = B[: len(B) // 10]
    torch.manual_seed(seed); vic = pretrain_source_model(X17[A], y17[A], dim)
    sur = {}
    for name, ix, off in (("same_A", A, 1000), ("disjoint_B", B, 2000), ("small_B10", B10, 3000)):
        torch.manual_seed(seed + off); sur[name] = pretrain_source_model(X17[ix], y17[ix], dim)
    pre = score_buffer(vic, X17[A], "logit"); sv = scores_of(vic, X18, "logit")
    ben = np.where(y18 == 0)[0]; att = np.where(y18 == 1)[0]
    top_v = ben[sv[ben] >= np.quantile(sv[ben], 0.99)]
    pools = {"random": ben}
    for name, m in sur.items():
        t = scores_of(m, X18, "logit"); pools[name] = ben[t[ben] >= np.quantile(t[ben], 0.99)]
    for name, dp in pools.items():
        hit = float(np.isin(dp, top_v).mean())
        for rate in (0.0, 0.01, 0.05):
            for j in range(8):
                rng = np.random.RandomState(seed * 1000 + j)
                n_att = int(round(N * 0.01)); n_dec = int(round(N * rate)); n_ben = N - n_att - n_dec
                idx = np.concatenate([rng.choice(ben, n_ben), rng.choice(att, n_att), rng.choice(dp, n_dec) if n_dec else np.array([], dtype=int)])
                rng.shuffle(idx)
                r = real_attack_metrics(aqt_preds(sv[idx], pre, 500, 0.99, 0.0), y18[idx])
                out.setdefault(f"{name}|{rate}", []).append(dict(seed=seed, hit=hit, **r))
    print("seed", seed, "done", flush=True)
json.dump(out, open("round8_surrogate.json", "w"))
import pandas as pd
rows = [dict(attacker=k.split("|")[0], decoy_rate=float(k.split("|")[1]), recall=np.mean([r["recall"] if "recall" in r else list(r.values())[0] for r in v]), hit=np.mean([r["hit"] for r in v])) for k, v in out.items()]
print(pd.DataFrame(rows).round(3).to_string(index=False))
