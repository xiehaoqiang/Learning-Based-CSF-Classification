"""I/O utilities for HORIBA/LabSpec SERSome exports.

A *SERSome* file is a tab-separated text export in which
    line 1          -> Raman shift axis (n_wavenumbers values)
    lines 2..N+1    -> N single spectra, one per line
The parser below is the one used for all analyses in the paper.
"""

from __future__ import annotations

import math
import os

import numpy as np


def read_horiba_txt(path: str, begin: int = 0, end: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read one SERSome export.

    Returns
    -------
    shift : (n_wavenumbers,) Raman shift axis in cm^-1
    specs : (n_spectra, n_wavenumbers) intensity matrix
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()

    n_lines = raw.count("\n")
    values = np.fromstring(raw, dtype=float, sep="\t")
    resolution = math.ceil(len(values) / n_lines)

    if (len(values) - resolution + 1) % resolution == 0:
        shift, specs = values[: resolution - 1], values[resolution - 1 :].reshape(-1, resolution)[:, 1:]
    elif (len(values) - resolution + 2) % resolution == 0:
        shift, specs = values[: resolution - 2], values[resolution - 2 :].reshape(-1, resolution)[:, 2:]
    else:
        shift, specs = values[:resolution], values[resolution:].reshape(-1, resolution)

    return shift[begin:end], specs[:, begin:end]


def build_sersome(data_dir: str, begin: int = 0, end: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load every .txt SERSome in ``data_dir``.

    Files are sorted naturally (CSF1, CSF2, ..., CSF10) so that the row order is
    deterministic and matches ``data/data_id_label.xlsx`` of the original project.

    Returns
    -------
    shift : (n_wavenumbers,)
    specs : (n_samples, n_spectra, n_wavenumbers)
    ids   : (n_samples,) integer sample ids parsed from the "CSFn-..." file names
    """
    names = [f for f in os.listdir(data_dir) if f.lower().endswith(".txt")]
    names.sort(key=_natural_key)

    specs, shift, ids = [], None, []
    for name in names:
        shift, one = read_horiba_txt(os.path.join(data_dir, name), begin=begin, end=end)
        specs.append(one)
        ids.append(_sample_id(name))

    return shift, np.asarray(specs, dtype=np.float64), np.asarray(ids, dtype=int)


def _natural_key(name: str) -> list:
    """Sort key that turns digit runs into integers (CSF2 < CSF10)."""
    out, num = [], ""
    for ch in name:
        if ch.isdigit():
            num += ch
        else:
            if num:
                out.append((1, int(num)))
                num = ""
            out.append((0, ch.lower()))
    if num:
        out.append((1, int(num)))
    return out


def _sample_id(name: str) -> int:
    """'CSF23-2-30min...' -> 23; falls back to 0 when no number is present."""
    digits = ""
    for ch in name:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    return int(digits) if digits else 0
