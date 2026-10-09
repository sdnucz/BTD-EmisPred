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
| `LICENSE` | Define the BSD 3-Clause license for the source code. |
| `.gitignore` | Exclude generated outputs, local configuration files and temporary files from Git tracking. |
| `README.md` | Describe the project, files and usage. |

## Usage

### 1. Install

Use Python 3.12 and install the dependencies:

```bash
conda create -n nir2-emispred python=3.12 -y
conda activate nir2-emispred
pip install -r requirements.txt
```

### 2. Train

Training data require `SMILES`, `Solvent` and `λem (nm)` columns.

Create a JSON configuration named `settings.json` with these required keys:

- Data splitting: `test_size`, `outer_folds`, `inner_folds`, `stratify_bins`.
- Feature construction: `morgan_radius`, `morgan_bits`.
- Feature selection: `selected_features`, `correlation_threshold`,
  `rfe_estimators`, `rfe_depth`, `rfe_step`.
- Parameter search: `n_trials`, `search_spaces`.

Optional `seed`, `threads` and `outer_jobs` control reproducibility and CPU use.
`search_spaces` maps candidate names (`XGB`, `CAT`, `LGBM`, `RF`, `GBR`,
`KNN`, `KRR`, `SVR`) to estimator parameters. Each parameter can be a fixed
value or a search specification: `kind` = `int` / `float` with `low`, `high`
and optional `step` / `log`, or `kind` = `categorical` with `choices`.

```bash
python train.py --data data/data/nir2_emission_dataset.csv \
  --config settings.json --output outputs/run
```

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
