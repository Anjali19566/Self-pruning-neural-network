"""
model.py
--------
Contains:
  - PrunableLinear  : custom linear layer with learnable sigmoid gates
  - AdaptiveSparseNet : full feed-forward network built from PrunableLinear blocks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ──────────────────────────────────────────────────────────────────
# PrunableLinear
# ──────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with per-weight sigmoid gates.

    Forward pass:
        gates          = sigmoid(gate_scores)        ∈ (0, 1)
        pruned_weights = weight ⊙ gates              element-wise
        output         = F.linear(x, pruned_weights, bias)

    Both `weight` and `gate_scores` are nn.Parameters — the optimiser
    updates both, and gradients flow through each independently.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias (same init as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Gate scores — same shape as weight.
        # Init to 2.0 → sigmoid(2) ≈ 0.88, so gates start mostly OPEN.
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), 2.0)
        )

        self._reset_weight()

    def _reset_weight(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates          = torch.sigmoid(self.gate_scores)   # (0, 1)
        pruned_weights = self.weight * gates               # element-wise
        return F.linear(x, pruned_weights, self.bias)

    # ── sparsity helpers ──────────────────────────────────────────

    def sparsity_loss(self) -> torch.Tensor:
        """L1 of all gate values. Always positive; encourages gates → 0."""
        return torch.sigmoid(self.gate_scores).sum()

    def sparsity_percent(self, threshold: float = 0.01) -> float:
        """% of weights whose gate is below `threshold` (effectively pruned)."""
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            return (gates < threshold).float().mean().item() * 100.0

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# ──────────────────────────────────────────────────────────────────
# AdaptiveSparseNet
# ──────────────────────────────────────────────────────────────────

class AdaptiveSparseNet(nn.Module):
    """
    Three-block feed-forward classifier for CIFAR-10 (input 3×32×32 = 3072).

    Each block: PrunableLinear → BatchNorm1d → GELU → Dropout
    Final layer: PrunableLinear (10 classes, no BN/activation)
    """

    def __init__(self, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.flatten = nn.Flatten()

        def block(d_in, d_out):
            return nn.Sequential(
                PrunableLinear(d_in, d_out),
                nn.BatchNorm1d(d_out),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.block0     = block(3072,       hidden)
        self.block1     = block(hidden,     hidden)
        self.block2     = block(hidden,     hidden // 2)
        self.classifier = PrunableLinear(hidden // 2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)

    # ── network-level sparsity helpers ────────────────────────────

    def _prunable_layers(self):
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """Sum of all gate values across every PrunableLinear in the network."""
        return sum(layer.sparsity_loss() for layer in self._prunable_layers())

    def overall_sparsity(self, threshold: float = 0.01) -> float:
        """Network-wide % of weights that are effectively pruned."""
        layers = self._prunable_layers()
        total  = sum(l.weight.numel() for l in layers)
        pruned = sum(
            int(l.weight.numel() * l.sparsity_percent(threshold) / 100)
            for l in layers
        )
        return pruned / total * 100.0

    def all_gate_values(self) -> np.ndarray:
        """Collect every gate value as a flat numpy array (for plotting)."""
        parts = []
        for layer in self._prunable_layers():
            with torch.no_grad():
                parts.append(
                    torch.sigmoid(layer.gate_scores).cpu().numpy().ravel()
                )
        return np.concatenate(parts)
