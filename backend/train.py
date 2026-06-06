"""
训练 LeNet-5

用法:
    python -m backend.train --epochs 5 --batch 64 --lr 0.05

权重会保存到 models/lenet5.npz
"""

import argparse
import os
import time
import numpy as np

from .model import LeNet5
from .data import load_mnist_sklearn, iterate_minibatches
from .core import Momentum


def evaluate(model, X, y, batch=256):
    """在 (X, y) 上跑一遍，返回 (accuracy, loss)."""
    total = 0
    correct = 0
    total_loss = 0.0
    n_batches = 0
    for xb, yb in iterate_minibatches(X, y, batch, shuffle=False):
        logits = model.forward(xb)
        loss = model.loss_fn.forward(logits, yb)
        preds = np.argmax(logits, axis=-1)
        correct += (preds == yb).sum()
        total += len(yb)
        total_loss += loss
        n_batches += 1
    return correct / total, total_loss / n_batches


def train(epochs=5, batch=64, lr=0.05, momentum=0.9,
          save_path="models/lenet5.npz", subset=None, seed=0):
    np.random.seed(seed)
    print(f"Config: epochs={epochs} batch={batch} lr={lr} momentum={momentum}")

    X_tr, y_tr, X_te, y_te = load_mnist_sklearn()

    if subset is not None:
        print(f"Using subset of {subset} training samples for speed.")
        idx = np.random.choice(len(X_tr), subset, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")

    model = LeNet5(use_sparse_c3=True)
    optimizer = Momentum(lr=lr, momentum=momentum)

    for ep in range(epochs):
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        for i, (xb, yb) in enumerate(iterate_minibatches(X_tr, y_tr, batch)):
            logits = model.forward(xb)
            loss = model.loss_fn.forward(logits, yb)
            dlogits = model.loss_fn.backward()
            model.backward(dlogits)
            optimizer.step(model._trainable_layers())
            running_loss += loss
            n_batches += 1
            if (i + 1) % 50 == 0:
                print(f"  epoch {ep+1} batch {i+1}/{len(X_tr)//batch} "
                      f"loss {running_loss/n_batches:.4f}")
        tr_acc, tr_loss = evaluate(model, X_tr, y_tr)
        te_acc, te_loss = evaluate(model, X_te, y_te)
        dt = time.time() - t0
        print(f"Epoch {ep+1}/{epochs} ({dt:.1f}s)  "
              f"train acc {tr_acc*100:.2f}% loss {tr_loss:.4f}  |  "
              f"test acc {te_acc*100:.2f}% loss {te_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Saved weights to {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--subset", type=int, default=None,
                        help="只用 N 个样本训练（调试用）")
    parser.add_argument("--save", type=str, default="models/lenet5.npz")
    args = parser.parse_args()
    train(epochs=args.epochs, batch=args.batch, lr=args.lr,
          momentum=args.momentum, subset=args.subset, save_path=args.save)
