# Learning-Based CSF Classification

Code for the convolutional neural network that discriminates adamantinomatous (ACP)
and papillary (PCP) craniopharyngioma from cerebrospinal-fluid SERSomes.

## Contents

| Path | What it is |
| --- | --- |
| `src/spectra_io.py` | Parser for the HORIBA SERSome `.txt` exports |
| `src/build_sersome.py` | Assembles the raw exports into the SERSome tensor |
| `src/model.py` | The CNN (12,409 parameters) and checkpoint loader |
| `src/train_cnn.py` | Training script |
| `src/predict.py` | Evaluation with the released weights |
| `src/shap_analysis.py` | SHAP interpretation of the trained CNN |
| `models/cnn_cp_subtyping.pt` | Weights used in the manuscript |
| `data/sample_labels.csv` | 62 samples: anonymised id, cohort, label (0 = ACP, 1 = PCP) |
| `notebooks/01_cnn_subtyping.ipynb` | Figure-oriented walkthrough of training and evaluation |

Raw SERSome exports are not stored in this repository (62 files, ~198 MB). Point
`--main-dir` / `--external-dir` at your local copies, or use the Zenodo data record.

## Installation

```bash
git clone https://github.com/xiehaoqiang/Learning-Based-CSF-Classification.git
cd Learning-Based-CSF-Classification
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 1.13+; CPU is sufficient.

## Usage

```bash
# 1. Build the SERSome tensor (one header line of Raman shifts + 200 spectra per file)
python src/build_sersome.py \
    --main-dir     /path/to/data_56 \
    --external-dir /path/to/data_unknown \
    --labels data/sample_labels.csv \
    --out ../sersome.npz

# 2. Reproduce the published predictions (Figure 4D)
python src/predict.py --npz ../sersome.npz --model models/cnn_cp_subtyping.pt

# 3. Optional: retrain from scratch (stochastic)
python src/train_cnn.py --npz ../sersome.npz --out models/cnn_cp_subtyping.pt

# 4. Optional: SHAP interpretation (Figure 6C-E)
python src/shap_analysis.py --npz ../sersome.npz --model models/cnn_cp_subtyping.pt
```

The assembled tensor is about 90 MB and is written outside the repository
(`../sersome.npz`) on purpose, so that only source code is committed.

## Verified reproduction

`src/predict.py` with the released weights:

| Cohort | Samples | Correct |
| --- | --- | --- |
| Internal test set | 9 | 8 |
| External cohort (independent collection batch) | 6 | 6 |
| **Total** | **15** | **14 (93.3 %)** |

Sensitivity for PCP 4/5, specificity for ACP 10/10, matching the confusion matrix
of Figure 4D. Baseline correction and cosmic-ray removal were applied in LabSpec 6
before export, so the `.txt` files are the starting point of this repository.

## Cohort definition

`data/sample_labels.csv` lists each sample with an anonymised `sample_id`, the cohort
(`train` 38 / `validation` 9 / `test` 9 / `external` 6) and the label
(`0 = ACP`, `1 = PCP`). Identifiers are study numbers only and carry no personal
information.

## Citation and licence

Cite the manuscript (TODO: authors, title, journal, DOI) and the archived release
(`10.5281/zenodo.XXXXXXX`). Code released under CC BY 4.0 (see `LICENSE`).

Lead contact: TODO.
