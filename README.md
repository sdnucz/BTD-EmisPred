# BTD-EmisPred

BTD-EmisPred predicts the fluorescence emission wavelength of BTD-based
molecules from molecular SMILES and solvent information. It combines molecular
fingerprints, RDKit fragment counts and solvent features with machine-learning
regression. Training uses molecule-grouped nested cross-validation for feature
selection, algorithm comparison and parameter optimization.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Run nested cross-validation, select an algorithm and its parameters, fit the final model and evaluate test predictions. |
| `predict.py` | Predict emission wavelengths for new molecule–solvent records using a trained model. |
| `data/dataset.py` | Clean records, group molecules, construct features and perform feature selection within training folds. |
| `emission_project/utils.py` | Provide shared molecular, solvent, data-reading and evaluation utilities. |
| `data/data/nir2_emission_dataset.csv` | Store molecular structures, solvents, experimental emission wavelengths and source information. |
| `requirements.txt` | List the Python dependencies and their versions. |
| `configs/example.json` | Combine manuscript-specified settings with editable example defaults for other parameters. |
| `tests/test_training.py` | Check selection rules and separation of training and test labels. |

## Usage

### 1. Install

Use Python 3.12 and install the dependencies:

```bash
conda create -n your_env python=3.12 -y
conda activate your_env
pip install -r requirements.txt
```

### 2. Train

Training data require `SMILES`, `Solvent` and `λem (nm)` columns.

Run training with the supplied configuration:

```bash
python train.py --data data/data/nir2_emission_dataset.csv \
  --config configs/example.json --output outputs/run
```

The configuration uses the settings specified in the manuscript:

| Setting | Value |
| --- | --- |
| Molecule-grouped training/test split | Approximately 80% / 20% |
| Outer / inner cross-validation | 10 / 3 folds |
| Morgan fingerprint length | 2,048 bits |
| Absolute correlation threshold | 0.5 |
| Features retained by random-forest RFE | 82 |
| Optuna trials per algorithm per selection round | 50 |
| Candidate algorithms | XGBoost, CatBoost, LightGBM, RF, GBR, KNN, KRR, SVR |

Parameters not specified in the manuscript retain editable example defaults:
seed 42, four molecular-target stratification bins, Morgan radius 2, RF-RFE with
50 trees, maximum depth 8 and step 100, and the example parameter search ranges
in `search_spaces`. CPU defaults are two threads and one outer fold at a time.

Within each outer training fold, preprocessing and feature selection are fitted
on the inner training folds. The lowest mean inner-fold RMSE selects the algorithm
and parameters. The selected pipeline is refitted on the full outer training fold
to predict its outer test fold. After outer CV, the same inner selection procedure
is repeated on the full training set, followed by refitting and held-out testing.

Edit `configs/example.json`, or copy it to a separate JSON file and pass that
file with `--config`. The configuration contains:

- Data splitting: `test_size`, `outer_folds`, `inner_folds`, `stratify_bins`.
- Feature construction: `morgan_radius`, `morgan_bits`.
- Feature selection: `selected_features`, `correlation_threshold`,
  `rfe_estimators`, `rfe_depth`, `rfe_step`.
- Parameter search: `n_trials`, `search_spaces`.

`seed`, `threads` and `outer_jobs` control reproducibility and CPU use.
`search_spaces` maps candidate names (`XGB`, `CAT`, `LGBM`, `RF`, `GBR`,
`KNN`, `KRR`, `SVR`) to estimator parameters. Each parameter can be a fixed
value or a search specification: `kind` = `int` / `float` with `low`, `high`
and optional `step` / `log`, or `kind` = `categorical` with `choices`.

Training writes cross-validation predictions, evaluation metrics, search
records and the final model to `outputs/run`. The trained model and its saved
feature selector are stored in `outputs/run/final`. Use a new output directory
for each training run.

### 3. Predict

Prepare a CSV file with `SMILES` and `Solvent` columns, then run:

```bash
python predict.py --model outputs/run/final \
  --input new_molecules.csv --output new_predictions.csv
```

The output preserves the input columns and adds `prediction_nm` (predicted
emission wavelength in nm) and `unknown_solvent` (whether the solvent was
absent from the model's training vocabulary). The saved selector supplies
the feature order and fingerprint settings. Use a new output filename for
each prediction run.
