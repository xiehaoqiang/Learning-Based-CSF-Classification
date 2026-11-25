"""Reproduce the subtyping results of Figure 4D / Figure S7 with the released weights.

Usage
-----
python src/predict.py --npz data/sersome.npz --model models/cnn_cp_subtyping.pt \
    --out ../predictions.csv

Expected output: 8/9 on the internal test set, 6/6 on the external cohort,
i.e. 14/15 = 93.3% overall, matching the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import load_splits, to_cnn_input  # noqa: E402
from model import load_model  # noqa: E402


def predict(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        prob = model(torch.from_numpy(to_cnn_input(x))).squeeze(-1).numpy()
    return prob, (prob > 0.5).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/sersome.npz")
    ap.add_argument("--model", default="models/cnn_cp_subtyping.pt")
    ap.add_argument("--out", default="../predictions.csv")
    args = ap.parse_args()

    s = load_splits(args.npz)
    model = load_model(args.model)

    rows = []
    for name in ("test", "ext"):
        if f"{name}_X" not in s:
            continue
        prob, pred = predict(model, s[f"{name}_X"])
        truth = s[f"{name}_y"]
        rows += [
            {
                "cohort": "external" if name == "ext" else "test",
                "index": i,
                "label": int(t),
                "probability_PCP": round(float(p), 4),
                "prediction": int(q),
                "correct": int(t == q),
            }
            for i, (t, p, q) in enumerate(zip(truth, prob, pred))
        ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    k = sum(r["correct"] for r in rows)
    tp = sum(1 for r in rows if r["label"] == 1 and r["prediction"] == 1)
    fn = sum(1 for r in rows if r["label"] == 1 and r["prediction"] == 0)
    fp = sum(1 for r in rows if r["label"] == 0 and r["prediction"] == 1)
    tn = sum(1 for r in rows if r["label"] == 0 and r["prediction"] == 0)

    print(f"accuracy : {k}/{n} = {k / n:.4f}")
    print(f"sensitivity (PCP): {tp}/{tp + fn}")
    print(f"specificity (ACP): {tn}/{tn + fp}")
    print("confusion matrix (rows = truth ACP/PCP, cols = prediction ACP/PCP):")
    print(f"  [[{tn}, {fp}],\n   [{fn}, {tp}]]")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
