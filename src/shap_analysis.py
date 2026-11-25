"""SHAP interpretation of the CNN (Figure 6C-E).

Usage
-----
python src/shap_analysis.py --npz data/sersome.npz --model models/cnn_cp_subtyping.pt \
    --out results/shap_values_mean.csv

The SHAP values are computed on the logit output of the network (before the
sigmoid), using 20 randomly drawn training SERSomes as the background
distribution, exactly as in the original analysis. Only correctly classified
test / external samples are explained.
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


class LogitWrapper(torch.nn.Module):
    """Expose the pre-sigmoid output of :class:`SERSomeCNN`."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.fc1(self.model._features(x))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/sersome.npz")
    ap.add_argument("--model", default="models/cnn_cp_subtyping.pt")
    ap.add_argument("--out", default="results/shap_values_mean.csv")
    ap.add_argument("--n-background", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import shap

    s = load_splits(args.npz)
    model = load_model(args.model)

    x_eval = np.concatenate([s["test_X"], s["ext_X"]]) if "ext_X" in s else s["test_X"]
    y_eval = np.concatenate([s["test_y"], s["ext_y"]]) if "ext_X" in s else s["test_y"]

    with torch.no_grad():
        prob = model(torch.from_numpy(to_cnn_input(x_eval))).squeeze(-1).numpy()
    correct = (prob > 0.5).astype(int) == y_eval
    x_explain = x_eval[correct]
    print(f"explaining {correct.sum()}/{len(correct)} correctly classified samples")

    rng = np.random.default_rng(args.seed)
    background = torch.from_numpy(to_cnn_input(s["train_X"][rng.choice(len(s["train_X"]), args.n_background, replace=False)]))

    explainer = shap.GradientExplainer(LogitWrapper(model).eval(), background)
    values = explainer.shap_values(torch.from_numpy(to_cnn_input(x_explain)))
    values = np.asarray(values[0] if isinstance(values, list) else values)
    print(f"SHAP tensor: {values.shape}")

    importance = np.abs(values).reshape(-1, values.shape[-1]).mean(axis=0)
    mean_spectrum = x_explain.reshape(-1, x_explain.shape[-1]).mean(axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["raman_shift", "mean_abs_shap", "mean_intensity"])
        for z, v, m in zip(s["shift"], importance, mean_spectrum):
            w.writerow([f"{z:.6f}", f"{v:.6f}", f"{m:.6f}"])
    print(f"written -> {args.out}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        fig, ax = plt.subplots(figsize=(8, 5))
        pts = np.array([s["shift"], mean_spectrum]).T.reshape(-1, 1, 2)
        lc = LineCollection(
            np.concatenate([pts[:-1], pts[1:]], axis=1),
            cmap="viridis",
            array=importance,
            linewidth=2,
        )
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.set_ylabel("Intensity")
        fig.colorbar(lc, ax=ax, label="mean |SHAP|")
        png = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        print(f"written -> {png}")
    except ImportError:
        print("matplotlib not available, skipped figure")


if __name__ == "__main__":
    main()
