"""
utils.py
--------
Utility functions:
  - get_loaders       : CIFAR-10 train / test DataLoaders
  - lambda_schedule   : curriculum sparsity ramp (warmup + cosine)
  - evaluate          : test accuracy + sparsity level
"""

import math
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────

def get_loaders(batch_size: int = 128, num_workers: int = 2):
    """
    Returns (train_loader, test_loader) for CIFAR-10.
    Training split uses random crop + flip + colour jitter for augmentation.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    root     = "./data"
    train_ds = torchvision.datasets.CIFAR10(root, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256,        shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────
# Lambda schedule  (curriculum sparsity)
# ──────────────────────────────────────────────────────────────────

def lambda_schedule(epoch: int, total_epochs: int,
                    lam_target: float, warmup_frac: float = 0.3) -> float:
    """
    Curriculum sparsity schedule:
      - Epochs 0 … warmup_end-1 : λ = 0   (learn to classify first)
      - Epochs warmup_end … end : λ ramps from 0 → lam_target via cosine curve

    Why curriculum?
    Applying full sparsity pressure from epoch 1 causes the network to fight
    two objectives simultaneously before it has learned anything useful.
    Delaying the sparsity ramp gives the weights time to find a good basin,
    so the subsequent pruning removes genuinely unimportant connections.
    """
    warmup_end = int(total_epochs * warmup_frac)
    if epoch < warmup_end:
        return 0.0
    progress = (epoch - warmup_end) / max(total_epochs - warmup_end, 1)
    return lam_target * 0.5 * (1.0 - math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """
    Returns:
        accuracy  : test accuracy (%)
        sparsity  : % of weights with gate < 0.01 (network-wide)
    """
    model.eval()
    correct = 0
    total   = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits  = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += images.size(0)
    return {
        "accuracy" : correct / total * 100.0,
        "sparsity" : model.overall_sparsity(),
    }
