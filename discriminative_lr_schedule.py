"""
Discriminative Layer-Wise Learning Rate Schedule
=================================================
Implementation of Algorithm 2 from:
  "Closing the NIDS Cold-Start Gap: Adaptive Transfer Learning for
   Day-Zero Intrusion Detection in Imbalanced Network Environments"

The schedule governs how learning rates are applied to each layer group
of the pre-trained (32, 16) MLP during online adaptation to the target
network. It implements a two-phase strategy:

  Phase 1 (n_t <= 1000): Only the output layer is fine-tuned.
           Hidden layers are frozen (effective rate ≈ 0), preserving
           the source-domain representations.

  Phase 2 (n_t > 1000): Hidden layers are progressively unfrozen using
           a sigmoid weighting function ω(n_t) that scales their learning
           rate from ~0 (at n_t = x_0) up to 0.1η (as n_t → ∞).

Parameters (from paper):
  η   = base learning rate (e.g. 0.00138, from Optuna search)
  k   = sigmoid steepness = 0.001
  x_0 = sigmoid midpoint  = 3600 (derived from source-domain convergence)
"""

import math
import torch
import torch.nn as nn
from torch.optim import Adam


# ---------------------------------------------------------------------------
# Sigmoid weighting function ω(n_t)
# ---------------------------------------------------------------------------

def sigmoid_weight(n_t: int, k: float = 0.001, x_0: float = 3600.0) -> float:
    """
    Compute the sigmoid weighting ω(n_t) = 1 / (1 + exp(-k * (n_t - x_0))).

    At n_t << x_0  →  ω ≈ 0  (hidden layers effectively frozen)
    At n_t  = x_0  →  ω = 0.5
    At n_t >> x_0  →  ω ≈ 1  (full discriminative rate applied)

    Args:
        n_t: Current number of target-domain samples processed.
        k:   Steepness of the sigmoid transition (default 0.001).
        x_0: Midpoint sample count (default 3600).

    Returns:
        ω ∈ (0, 1)
    """
    return 1.0 / (1.0 + math.exp(-k * (n_t - x_0)))


# ---------------------------------------------------------------------------
# Layer-wise learning rate computation
# ---------------------------------------------------------------------------

def get_layer_rates(
    n_t: int,
    eta: float,
    k: float = 0.001,
    x_0: float = 3600.0,
    phase1_cutoff: int = 1000,
) -> dict:
    """
    Return the effective learning rate for each layer group at sample n_t.

    Phase 1 (n_t <= phase1_cutoff):
        output layer  → eta
        hidden layers → 0.0   (frozen)

    Phase 2 (n_t > phase1_cutoff):
        output layer  → eta
        hidden layers → 0.1 * eta * ω(n_t)

    Args:
        n_t:           Current target-domain sample count.
        eta:           Base learning rate.
        k:             Sigmoid steepness.
        x_0:           Sigmoid midpoint.
        phase1_cutoff: Sample count at which Phase 2 begins (default 1000).

    Returns:
        dict with keys 'output' and 'hidden', each mapping to a float lr.
    """
    omega = sigmoid_weight(n_t, k=k, x_0=x_0)

    if n_t <= phase1_cutoff:
        # Phase 1: output layer only
        return {
            "output": eta,
            "hidden": 0.0,
        }
    else:
        # Phase 2: discriminative unfreezing
        return {
            "output": eta,
            "hidden": 0.1 * eta * omega,
        }


# ---------------------------------------------------------------------------
# MLP definition  (32, 16) as described in the paper
# ---------------------------------------------------------------------------

class ATL_MLP(nn.Module):
    """
    Compact (32, 16) MLP for cold-start NIDS.
    Input: 14-dimensional PCA-reduced feature vector.
    Output: binary logit (Benign / Anomaly).
    ~600 parameters, ~1,200 FLOPS/inference.
    """

    def __init__(self, n_input: int = 14):
        super().__init__()
        self.hidden1 = nn.Linear(n_input, 32)
        self.hidden2 = nn.Linear(32, 16)
        self.output  = nn.Linear(16, 1)
        self.act     = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.hidden1(x))
        x = self.act(self.hidden2(x))
        return self.output(x)

    def output_params(self):
        """Parameters belonging to the output layer."""
        return list(self.output.parameters())

    def hidden_params(self):
        """Parameters belonging to the hidden layers."""
        return list(self.hidden1.parameters()) + list(self.hidden2.parameters())


# ---------------------------------------------------------------------------
# Optimizer factory — builds a param-group Adam with per-layer rates
# ---------------------------------------------------------------------------

def build_optimizer(
    model: ATL_MLP,
    n_t: int,
    eta: float,
    k: float = 0.001,
    x_0: float = 3600.0,
    phase1_cutoff: int = 1000,
) -> Adam:
    """
    Construct an Adam optimizer whose learning rates reflect Algorithm 2.

    Because PyTorch optimizers do not support dynamic per-group lr updates
    natively inside the optimizer step, the caller should rebuild or update
    the optimizer's param groups at each sample step (or each mini-batch).
    See `update_optimizer_lr` for in-place lr updates without rebuilding.

    Args:
        model:         ATL_MLP instance.
        n_t:           Current target-domain sample count.
        eta:           Base learning rate.
        k, x_0:        Sigmoid parameters.
        phase1_cutoff: Phase boundary (default 1000).

    Returns:
        Adam optimizer with two parameter groups.
    """
    rates = get_layer_rates(n_t, eta, k=k, x_0=x_0, phase1_cutoff=phase1_cutoff)

    param_groups = [
        {"params": model.hidden_params(), "lr": rates["hidden"]},
        {"params": model.output_params(),  "lr": rates["output"]},
    ]
    return Adam(param_groups)


def update_optimizer_lr(
    optimizer: Adam,
    n_t: int,
    eta: float,
    k: float = 0.001,
    x_0: float = 3600.0,
    phase1_cutoff: int = 1000,
) -> None:
    """
    Update the learning rates of an existing optimizer in-place.

    Param group 0 → hidden layers
    Param group 1 → output layer

    This avoids rebuilding the optimizer (which would reset momentum state)
    on every sample step.
    """
    rates = get_layer_rates(n_t, eta, k=k, x_0=x_0, phase1_cutoff=phase1_cutoff)
    optimizer.param_groups[0]["lr"] = rates["hidden"]
    optimizer.param_groups[1]["lr"] = rates["output"]


# ---------------------------------------------------------------------------
# Online adaptation loop — wires Algorithm 2 into a streaming update
# ---------------------------------------------------------------------------

def adapt_stream(
    model: ATL_MLP,
    stream,                      # iterable of (x_tensor, y_tensor) pairs
    eta: float = 0.00138,
    k: float = 0.001,
    x_0: float = 3600.0,
    phase1_cutoff: int = 1000,
    loss_fn=None,
):
    """
    Adapt the pre-trained model to an incoming target-domain stream,
    applying Algorithm 2 at every sample step.

    Args:
        model:         Pre-trained ATL_MLP.
        stream:        Iterable yielding (x, y) tensors one sample at a time.
        eta:           Base learning rate (0.00138 from Optuna).
        k, x_0:        Sigmoid parameters.
        phase1_cutoff: Phase 1 / Phase 2 boundary.
        loss_fn:       Loss function (default: BCEWithLogitsLoss).

    Yields:
        (n_t, loss_value, lr_hidden, lr_output) for each sample processed.
    """
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    optimizer = build_optimizer(
        model, n_t=0, eta=eta, k=k, x_0=x_0, phase1_cutoff=phase1_cutoff
    )

    model.train()
    for n_t, (x, y) in enumerate(stream, start=1):

        # ── Algorithm 2: update layer-wise learning rates ──────────────────
        update_optimizer_lr(optimizer, n_t, eta, k=k, x_0=x_0,
                            phase1_cutoff=phase1_cutoff)

        # ── Gradient step ──────────────────────────────────────────────────
        optimizer.zero_grad()
        logit = model(x.unsqueeze(0) if x.dim() == 1 else x)
        loss  = loss_fn(logit.squeeze(), y.float())
        loss.backward()
        optimizer.step()

        rates = get_layer_rates(n_t, eta, k=k, x_0=x_0,
                                phase1_cutoff=phase1_cutoff)
        yield n_t, loss.item(), rates["hidden"], rates["output"]


# ---------------------------------------------------------------------------
# Quick demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("Algorithm 2: Discriminative Layer-Wise Learning Rate Schedule")
    print("=" * 62)
    print(f"{'n_t':>8}  {'Phase':>7}  {'ω(n_t)':>8}  {'lr_hidden':>12}  {'lr_output':>12}")
    print("-" * 62)

    eta   = 0.00138   # from Optuna search
    k     = 0.001
    x_0   = 3600.0

    checkpoints = [50, 100, 500, 1000, 1001, 2000, 3600, 5000, 7200, 10000]
    for n_t in checkpoints:
        omega = sigmoid_weight(n_t, k=k, x_0=x_0)
        rates = get_layer_rates(n_t, eta, k=k, x_0=x_0)
        phase = "1 (frozen)" if n_t <= 1000 else "2 (unfreeze)"
        print(f"{n_t:>8}  {phase:>11}  {omega:>8.4f}  "
              f"{rates['hidden']:>12.6f}  {rates['output']:>12.6f}")

    print()
    print("MLP parameter count:")
    model = ATL_MLP(n_input=14)
    total = sum(p.numel() for p in model.parameters())
    # Full count with bias terms: 14*32+32 + 32*16+16 + 16*1+1 = 1025
    # The paper's "~600" is approximate and refers to the weight matrices only
    # (14*32 + 32*16 + 16*1 = 976 weights), or a slightly condensed representation.
    weights_only = sum(p.numel() for p in model.parameters() if p.dim() > 1)
    print(f"  Total (weights + bias): {total}")
    print(f"  Weights only:           {weights_only}")
