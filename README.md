# Self-Pruning Neural Network

🚀 A neural network that **learns to prune its own weights during training**, reducing model size while preserving performance—eliminating the need for a separate pruning stage.

---

## 📌 Overview

Deep neural networks are often heavily **over-parameterized**, leading to unnecessary computation and memory usage.
This project introduces a **self-pruning mechanism** where each weight is assigned a learnable gate that determines its importance during training.

* Important connections are preserved
* Less useful connections are gradually suppressed
* The network automatically becomes more compact over time

---

## 🧠 Core Idea

Each weight is modulated by a learnable gate:

```python
gate = sigmoid(gate_score)
weight = weight * gate
```

### Loss Function

```python
loss = classification_loss + λ * L1(gates)
```

* Gate → **0** ⇒ connection is pruned
* Gate → **1** ⇒ connection is retained

---

## 🏗️ Architecture

```
Input (CIFAR-10)
   ↓
Flatten
   ↓
FC (1024) → BatchNorm → ReLU
   ↓
FC (512)  → BatchNorm → ReLU
   ↓
FC (256)  → BatchNorm → ReLU
   ↓
FC (10)
```

All layers use a custom **PrunableLinear** module with learnable gating.

---

## 📊 Results

### 🔹 Gate Distribution

![Gate Distribution](results/plots.png)

---

## 📁 Project Structure

```
model.py    # PrunableLinear layer + model definition
train.py    # Training loop and experiment runner
utils.py    # Data handling and evaluation utilities
plots.py    # Visualization scripts
results/    # Generated outputs (plots, models)
```

---

## ⚙️ Tech Stack

* PyTorch
* NumPy
* Matplotlib

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python train.py
```

Run experiments with different sparsity levels:

```bash
python train.py --lambdas 0.0001 0.001 0.01 --epochs 60
```

---

## 🔍 Key Highlights

* Dynamic pruning integrated directly into training
* No separate pruning or fine-tuning pipeline required
* Learnable sparsity using L1 regularization on gates
* Trade-off exploration between accuracy and efficiency

---

## 🔮 Future Work

* Extend the method to **convolutional neural networks** for improved visual performance
* Investigate **structured pruning approaches** (channel-level and filter-level compression)
* Explore advanced sparsity techniques such as **L0 regularization and Hard Concrete gating** for stronger compression
* Improve sparsity scheduling strategies to encourage more effective gate suppression during training

---

## ⭐ Summary

This project demonstrates how neural networks can **adaptively learn their own compact structure during training**, making them more efficient without requiring post-processing pruning steps.
