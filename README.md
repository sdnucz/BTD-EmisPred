# BTD-EmisPred

## Purpose

Predict fluorescence emission wavelengths of BTD-based molecules from SMILES
and solvent information using machine learning. The training workflow uses
molecule-grouped nested cross-validation for feature selection and comparison
of eight regression algorithms.

Use the trained model on the [website](https://btd-emispred.online), or download
the code and model from [Zenodo (project DOI)](https://doi.org/10.5281/zenodo.24186343).

**Archive notice:** The Zenodo record cited in the manuscript
(DOI: [10.5281/zenodo.24186344](https://doi.org/10.5281/zenodo.24186344))
was inadvertently deleted during archive consolidation. A restoration request
has been submitted to Zenodo. Please use the project DOI above to access the
current archive.

## Files

| File or directory | Purpose |
| --- | --- |
| `train.py`, `reproduce.py` | Train, select and evaluate models using the configured searches and fixed folds. |
| `predict.py` | Predict emission wavelengths for new molecule–solvent records. |
| `pretrained/` | Trained LightGBM model, selected features, preprocessing information and model provenance. |
| `configs/paper.json` | Training settings and algorithm search spaces. |
| `data/data/nir2_emission_dataset.csv` | Molecular structures, solvents, emission wavelengths and sources. |
| `data/dataset.py`, `emission_project/utils.py` | Data processing, feature construction, feature selection and evaluation utilities. |
| `splits/`, `frozen_partitions.py` | Fixed training/test and nested-CV partitions, with integrity checks. |
| `reference/` | Saved configurations, parameters, predictions and figure source data. |
| `refit_reference.py`, `compare_reproduction.py` | Refit saved parameters and compare with reference results. |
| `verify_reference.py`, `verify_run.py` | Check supplied model/results or a completed training run. |
| `plot_review_results.py` | Draw Supplementary Figures 2–3 and ≥1000 nm OOF/test comparisons; export source tables and outer-fold algorithm selections. |
| `count_similarity.py`, `plot_count_similarity.py` | Calculate count-Morgan fingerprint similarity and draw the shaded lollipop plot. |
| `data/similarity_queries.csv` | Structures and labels for the 13 similarity-query molecules. |
| `requirements.txt`, `tests/` | Dependencies and automated checks. |

## Usage

Run commands from the repository root. Use a new output directory for each run.

### Install

```bash
conda create -n your_env python=3.12 -y
conda activate your_env
pip install -r requirements.txt
```

### Train

The supplied configuration uses 10 outer folds, 3 inner folds, 82 selected
features and 50 Optuna trials per algorithm. Settings are in `configs/paper.json`.

```bash
python reproduce.py --output outputs/run
```

This runs training and completion checks. The final model is saved in
`outputs/run/final`.

### Predict

Prepare a CSV containing `SMILES` and `Solvent` columns:

```bash
python predict.py --model pretrained \
  --input new_molecules.csv --output new_predictions.csv
```

For a newly trained model, replace `pretrained` with `outputs/run/final`.
The output adds `prediction_nm` and an `unknown_solvent` flag.

### Generate comparison figures

Check the supplied reference results and generate figures without retraining:

```bash
python verify_reference.py --output outputs/reference_check --figures
```

For a new training run, keep `export_candidate_predictions` enabled and use:

```bash
python plot_review_results.py --run outputs/run --output outputs/review_figures
```

Outputs include eight-algorithm OOF scatter plots, fold-level RMSE/MAE/R²
comparisons, and ≥1000 nm nested-OOF/test MAE and RMSE with record/molecule counts.
Figures are exported as PDF, SVG and PNG, alongside CSV source tables.

### Calculate and plot structural similarity

```bash
python count_similarity.py --output outputs/count_similarity
python plot_count_similarity.py --input outputs/count_similarity/Count_similarity.csv \
  --output outputs/count_similarity/figures
```

This calculates maximum Tanimoto similarity to the fixed training structures
using sparse-count Morgan fingerprints (radius 2), and exports the pale-background
lollipop plot and source data.
