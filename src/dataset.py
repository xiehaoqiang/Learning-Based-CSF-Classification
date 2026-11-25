"""Splitting / reshaping helpers shared by the training and evaluation scripts."""

from __future__ import annotations

import numpy as np


def load_splits(npz_path: str) -> dict[str, np.ndarray]:
    """Return train/validation/test (and external when available) arrays.

    X arrays have shape (n_samples, n_spectra, n_wavenumbers); the channel
    dimension expected by the CNN is added by :func:`to_cnn_input`.
    """
    d = np.load(npz_path, allow_pickle=False)
    cohorts = d["cohorts"]
    out = {
        "shift": d["shift"],
        "train_X": d["X"][cohorts == "train"],
        "train_y": d["labels"][cohorts == "train"],
        "valid_X": d["X"][cohorts == "validation"],
        "valid_y": d["labels"][cohorts == "validation"],
        "test_X": d["X"][cohorts == "test"],
        "test_y": d["labels"][cohorts == "test"],
    }
    if "ext_X" in d:
        out["ext_X"] = d["ext_X"]
        out["ext_y"] = d["ext_labels"]
    return out


def to_cnn_input(x: np.ndarray) -> np.ndarray:
    """(n, n_spectra, n_wavenumbers) -> (n, 1, n_spectra, n_wavenumbers) float32."""
    return np.asarray(x, dtype=np.float32)[:, None, :, :]
