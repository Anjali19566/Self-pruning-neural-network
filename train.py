"""
train.py
--------
Entry point. Trains AdaptiveSparseNet on CIFAR-10 across multiple λ values,
saves results to results/ and prints a final summary table.

Usage
-----
  python train.py                                         # default λ sweep
  python train.py --lambdas 0.0001 0.001 0.01            # custom sweep
  python train.py --lambdas 0.0005 --epochs 60           # single λ, more epochs
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from model import AdaptiveSparseNet
from utils import get_loaders, lambda_schedule, evaluate
from plots import plot_gate_histogram, plot_tradeoff


# ──────────────────────────────────────────────────────────────────
# Single training epoch
# ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, lam: float) -> dict:
    model.train()
    total_cls  = 0.0
    total_spar = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits     = model(images)
        cls_loss   = nn.functional.cross_entropy(logits, labels)
        spar_loss  = model.sparsity_loss()
        loss       = cls_loss + lam * spar_loss

        optimizer.zero_grad()
        loss.backward()
        # Clip gradients — gate_scores can get large early in training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_cls  += cls_loss.item()  * images.size(0)
        total_spar += spar_loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)

    n = len(loader.dataset)
    return {
        "cls_loss" : total_cls  / n,
        "spar_loss": total_spar / n,
        "train_acc": correct / total * 100.0,
    }


# ──────────────────────────────────────────────────────────────────
# Single experiment  (one λ value)
# ──────────────────────────────────────────────────────────────────

def run_experiment(lam: float, epochs: int, lr: float, batch_size: int,
                   device: torch.device, out_dir: Path) -> dict:

    print(f"\n{'='*60}")
    print(f"  Experiment  λ = {lam}")
    print(f"{'='*60}")

    train_loader, test_loader = get_loaders(batch_size=batch_size)
    model = AdaptiveSparseNet(hidden=512, dropout=0.3).to(device)

    # Give gate_scores a 2× higher LR — they are binary decisions and
    # need to move faster than the continuous weight parameters.
    gate_params   = [p for n, p in model.named_parameters() if "gate_scores" in n]
    weight_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]

    optimizer = torch.optim.Adam([
        {"params": weight_params, "lr": lr},
        {"params": gate_params,   "lr": lr * 2},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        lam_now = lambda_schedule(epoch, epochs, lam)
        tr      = train_one_epoch(model, train_loader, optimizer, device, lam_now)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            ev = evaluate(model, test_loader, device)
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"cls={tr['cls_loss']:.4f}  "
                  f"λ_eff={lam_now:.5f}  "
                  f"test_acc={ev['accuracy']:.2f}%  "
                  f"sparsity={ev['sparsity']:.1f}%")

    final = evaluate(model, test_loader, device)
    print(f"\n  ✓  accuracy={final['accuracy']:.2f}%   sparsity={final['sparsity']:.1f}%")

    # Save gate histogram
    gate_vals = model.all_gate_values()
    plot_gate_histogram(gate_vals, lam, str(out_dir / "plots.png"))

    # Save model checkpoint
    torch.save(model.state_dict(), str(out_dir / f"model_lambda_{lam}.pt"))

    return {
        "lambda"   : lam,
        "accuracy" : final["accuracy"],
        "sparsity" : final["sparsity"],
        "gate_vals": gate_vals.tolist(),
    }


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Self-Pruning Network – CIFAR-10")
    p.add_argument("--lambdas",    nargs="+", type=float,
                   default=[1e-4, 5e-4, 1e-3, 5e-3],
                   help="λ values to sweep  (default: 4 values)")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args    = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print(f"\nDevice  : {device}")
    print(f"Lambdas : {args.lambdas}")
    print(f"Epochs  : {args.epochs}")

    all_results = []
    for lam in args.lambdas:
        res = run_experiment(
            lam        = lam,
            epochs     = args.epochs,
            lr         = args.lr,
            batch_size = args.batch_size,
            device     = device,
            out_dir    = out_dir,
        )
        all_results.append(res)

    # Trade-off plot
    plot_tradeoff(all_results, str(out_dir / "tradeoff.png"))

    # Markdown results table  →  results/table.md
    lines = [
        "| Lambda | Test Accuracy | Sparsity Level (%) |",
        "|:------:|:-------------:|:------------------:|",
    ]
    for r in all_results:
        lines.append(f"| {r['lambda']:.5f} | {r['accuracy']:.2f}% | {r['sparsity']:.1f}% |")
    (out_dir / "table.md").write_text("\n".join(lines))

    # Save raw JSON
    summary = [{"lambda": r["lambda"], "accuracy": r["accuracy"],
                 "sparsity": r["sparsity"]} for r in all_results]
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print final table
    print("\n" + "="*52)
    print(f"{'Lambda':>12}  {'Test Accuracy':>15}  {'Sparsity':>10}")
    print("-"*52)
    for r in all_results:
        print(f"{r['lambda']:>12.5f}  {r['accuracy']:>14.2f}%  {r['sparsity']:>9.1f}%")
    print("="*52)
    print(f"\nOutputs saved to: {out_dir}/\n")


if __name__ == "__main__":
    main()
