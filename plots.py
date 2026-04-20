"""
plots.py
--------
All plotting functions:
  - plot_gate_histogram : distribution of final gate values for one model
  - plot_tradeoff       : accuracy vs. sparsity across λ values
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ──────────────────────────────────────────────────────────────────
# Gate histogram
# ──────────────────────────────────────────────────────────────────

def plot_gate_histogram(gate_values: np.ndarray, lam: float, out_path: str):
    """
    Histogram of all σ(gate_score) values for a trained model.

    A successful result shows:
      - A large spike near 0  →  pruned (dead) weights
      - A smaller cluster near 1 →  active weights
    This bimodal pattern confirms the gates have made decisive choices.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    n_pruned = int((gate_values < 0.01).sum())
    n_total  = len(gate_values)
    pct      = n_pruned / n_total * 100

    ax.hist(gate_values, bins=120, color="#2563EB", edgecolor="none", alpha=0.85)
    ax.axvline(0.01, color="#DC2626", linestyle="--", linewidth=1.8,
               label=f"prune threshold = 0.01  ({pct:.1f}% pruned)")

    ax.set_xlabel("Gate value  σ(gate_score)", fontsize=12)
    ax.set_ylabel("Count",                     fontsize=12)
    ax.set_title(f"Gate Value Distribution  (λ = {lam})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    # Annotate the two clusters
    ax.annotate("pruned\nweights", xy=(0.005, ax.get_ylim()[1] * 0.7),
                fontsize=10, color="#DC2626", ha="center")
    ax.annotate("active\nweights", xy=(0.85, ax.get_ylim()[1] * 0.5),
                fontsize=10, color="#16A34A", ha="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → Gate histogram saved: {out_path}")


# ──────────────────────────────────────────────────────────────────
# Trade-off plot
# ──────────────────────────────────────────────────────────────────

def plot_tradeoff(results: list, out_path: str):
    """
    Dual-axis plot: accuracy (blue) and sparsity (green) vs. λ.
    x-axis is log-scaled since λ values span orders of magnitude.
    """
    lambdas  = [r["lambda"]   for r in results]
    accs     = [r["accuracy"] for r in results]
    sparses  = [r["sparsity"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 4))

    col_acc = "#2563EB"
    col_sp  = "#16A34A"

    ax1.set_xlabel("λ  (sparsity weight, log scale)", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", color=col_acc, fontsize=12)
    ax1.plot(lambdas, accs, "o-", color=col_acc, linewidth=2.2,
             markersize=8, label="Test Accuracy")
    ax1.tick_params(axis="y", labelcolor=col_acc)
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Sparsity Level (%)", color=col_sp, fontsize=12)
    ax2.plot(lambdas, sparses, "s--", color=col_sp, linewidth=2.2,
             markersize=8, label="Sparsity")
    ax2.tick_params(axis="y", labelcolor=col_sp)

    # Label each point
    for lam, acc, sp in zip(lambdas, accs, sparses):
        ax1.annotate(f"{acc:.1f}%",  (lam, acc),  textcoords="offset points",
                     xytext=(0, 8),  fontsize=9, color=col_acc, ha="center")
        ax2.annotate(f"{sp:.1f}%",   (lam, sp),   textcoords="offset points",
                     xytext=(0, -14), fontsize=9, color=col_sp,  ha="center")

    ax1.set_title("λ Trade-off: Accuracy vs. Sparsity", fontsize=13, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → Trade-off plot saved: {out_path}")
