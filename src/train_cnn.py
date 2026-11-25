"""Train the SERSome CNN from scratch (Figure 4C-D).

Usage
-----
python src/train_cnn.py --npz data/sersome.npz --out models/cnn_cp_subtyping.pt

Hyper-parameters follow the STAR Methods ("The CNN architecture and occlusion
method"): Adam (lr 1e-4, weight decay 2e-5), StepLR (gamma 0.8, step 20),
batch size 8, up to 100 epochs, weighted binary cross-entropy.

Training is stochastic: re-running this script will not reproduce the exact
weights in ``models/cnn_cp_subtyping.pt``. Use ``src/predict.py`` with the
released weights to reproduce the published numbers.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import load_splits, to_cnn_input  # noqa: E402
from model import SERSomeCNN  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def weighted_bce(prob: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """Binary cross-entropy with an explicit positive-class weight (Methods)."""
    per_element = nn.functional.binary_cross_entropy(prob, target, reduction="none")
    weights = torch.where(target > 0.5, pos_weight, 1.0)
    return (per_element * weights).mean()


def accuracy(model: SERSomeCNN, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        pred = (model(torch.from_numpy(to_cnn_input(x))).squeeze(-1).numpy() > 0.5).astype(int)
    return float((pred == y).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/sersome.npz")
    ap.add_argument("--out", default="models/cnn_cp_subtyping.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--history", default="../training_history.csv")
    args = ap.parse_args()

    set_seed(args.seed)
    s = load_splits(args.npz)
    x_tr, y_tr = s["train_X"], s["train_y"]
    x_va, y_va = s["valid_X"], s["valid_y"]

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"train {x_tr.shape}, valid {x_va.shape}, pos_weight {pos_weight:.2f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SERSomeCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    xt = torch.from_numpy(to_cnn_input(x_tr))
    yt = torch.from_numpy(y_tr.astype(np.float32))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt, yt), batch_size=args.batch_size, shuffle=True
    )

    best_loss, best_state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = weighted_bce(model(xb).squeeze(-1), yb, pos_weight)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vl = weighted_bce(
                model(torch.from_numpy(to_cnn_input(x_va)).to(device)).squeeze(-1),
                torch.from_numpy(y_va.astype(np.float32)).to(device),
                pos_weight,
            ).item()

        train_loss = running / len(xt)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": vl})
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  valid {vl:.4f}")

        if vl < best_loss:
            best_loss, best_epoch = vl, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"state_dict": best_state, "pos_weight": pos_weight, "seed": args.seed}, args.out)
    model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(os.path.abspath(args.history)), exist_ok=True)
    with open(args.history, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "valid_loss"])
        w.writeheader()
        w.writerows(history)

    print(f"best valid loss {best_loss:.4f}")
    print(f"validation accuracy {accuracy(model, x_va, y_va):.4f}")
    print(f"test accuracy       {accuracy(model, s['test_X'], s['test_y']):.4f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
