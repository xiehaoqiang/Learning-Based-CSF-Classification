"""Build the SERSome tensor used by every downstream script.

Usage
-----
python src/build_sersome.py \
    --main-dir   /path/to/data_56 \
    --external-dir /path/to/data_unknown \
    --labels     data/sample_labels.csv \
    --out        data/sersome.npz

The raw .txt exports are not distributed with this repository; this script is the
only place that touches them.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spectra_io import build_sersome  # noqa: E402


def read_labels(path: str) -> dict[int, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {int(row["sample_id"]): row for row in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-dir", required=True, help="directory with the 56 in-house SERSome .txt files")
    ap.add_argument("--external-dir", default=None, help="directory with the 6 external-validation SERSome .txt files")
    ap.add_argument("--labels", default="data/sample_labels.csv")
    ap.add_argument("--out", default="data/sersome.npz")
    ap.add_argument("--begin", type=int, default=0, help="first wavenumber index to keep")
    ap.add_argument("--end", type=int, default=None, help="last wavenumber index to keep (exclusive)")
    args = ap.parse_args()

    shift, specs, ids = build_sersome(args.main_dir, args.begin, args.end)
    print(f"in-house SERSome: {specs.shape}  wavenumbers: {shift.size}")

    table = read_labels(args.labels)
    missing = [int(i) for i in ids if int(i) not in table]
    if missing:
        raise SystemExit(f"sample ids missing from {args.labels}: {missing}")

    labels = np.array([int(table[int(i)]["label"]) for i in ids], dtype=int)
    cohorts = np.array([table[int(i)]["cohort"] for i in ids])

    payload = {"shift": shift, "X": specs, "ids": ids, "labels": labels, "cohorts": cohorts}

    if args.external_dir:
        _, ext_specs, ext_ids = build_sersome(args.external_dir, args.begin, args.end)
        if ext_specs.shape[1:] != specs.shape[1:]:
            raise SystemExit(f"external shape {ext_specs.shape[1:]} != in-house {specs.shape[1:]}")
        ext_labels = np.array([int(table[int(i)]["label"]) for i in ext_ids], dtype=int)
        payload.update(ext_X=ext_specs, ext_ids=ext_ids, ext_labels=ext_labels)
        print(f"external SERSome: {ext_specs.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"saved -> {args.out}")

    for name in ("train", "validation", "test"):
        print(f"  {name}: {(cohorts == name).sum()} samples")
    if args.external_dir:
        print(f"  external: {len(ext_ids)} samples")


if __name__ == "__main__":
    main()
