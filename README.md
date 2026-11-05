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
| `configs/example.json` | Provide a complete example configuration for data splitting, feature selection and hyperparameter search across eight algorithms. |
| `configs/paper.json` | Define the manuscript workflow and the archived eight-algorithm search spaces. |
| `tests/test_training.py` | Check algorithm selection, prediction exports and separation of training and test labels. |

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

Run the manuscript workflow:

```bash
python train.py --data data/data/nir2_emission_dataset.csv \
  --config configs/paper.json --output outputs/paper
```

The supplied dataset contains 1,033 molecule–solvent records from 483 molecular
structures. With the paper configuration, molecule-grouped splitting assigns
834 records / 386 molecules to training and 199 records / 97 molecules to testing.
The workflow is:

1. Run molecule-grouped 10-fold outer CV on the training set.
2. Within each outer training fold, use shared molecule-grouped 3-fold inner CV
   for all eight candidate algorithms. Fit the solvent vocabulary, zero-variance
   filter, absolute Pearson-correlation filter (threshold 0.5) and random-forest
   RFE on each inner training fold. RFE retains 82 features. KNN, KRR and SVR
   additionally standardize their selected inputs using that training fold.
3. Run 50 Optuna trials per algorithm and select the algorithm and parameters
   with the lowest mean inner-fold RMSE. Refit preprocessing, feature selection
   and the selected algorithm on the full outer training fold, then predict its
   outer test fold. Pool the ten outer predictions to evaluate the selection
   procedure using R², RMSE and MAE.
4. Repeat the same 3-fold algorithm and parameter selection on all 834 training
   records. Refit the selected pipeline on these records and predict the
   independent test set. The final algorithm is determined by this search.

Feature construction uses radius-2, 2,048-bit Morgan fingerprints, 15 RDKit
fragment counts and training-fitted solvent one-hot features. The full training
set produces 32 solvent features; the vocabulary and selected feature names may
vary between folds. RF-RFE uses standardized candidate features; tree regressors
receive the selected features in their original scale. XGBoost uses squared-error
training loss. Exact algorithm-score ties follow the order XGB, CAT, LGBM, RF,
GBR, KNN, KRR, SVR.

`paper.json` records the manuscript's fold counts, feature count, correlation
threshold and search budget. Implementation settings, including seed 48,
five molecular-target stratification bins, RF-RFE settings (150 trees, depth 10,
step 20), and parameter ranges, follow the archived 82-feature training setup.
SVR is capped at 1,000,000 iterations; nonconvergent or invalid trials are recorded
as pruned. The configuration runs one outer fold at a time with four CPU threads.
There are 4,400 search trials across the ten outer folds and final selection.

For a smaller demonstration run:

```bash
python train.py --data data/data/nir2_emission_dataset.csv \
  --config configs/example.json --output outputs/run
```

The example uses molecule-grouped 3-fold outer and 3-fold inner cross-validation,
64 selected features and 3 Optuna trials per algorithm in each selection round.
It reserves 20% of molecular groups for testing, uses seed 42 and runs one outer
fold at a time with two CPU threads.

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

The paper configuration enables `export_candidate_predictions`. Training then
also refits each candidate with its own inner-selected parameters to produce
separate outer predictions for algorithm comparison. These predictions do not
enter the selection criterion.

Important outputs (paths are relative to the chosen output directory):

| Output | Use |
| --- | --- |
| `Split_Assignments.csv`, `CV_Partitions.json`, `outer/*/inner/*/Partition.json`, `final/inner/*/Partition.json` | Record the train/test, outer and inner record assignments. |
| `Nested_OOF_Predictions.csv` | Pooled outer predictions of the complete algorithm-selection procedure. |
| `Candidate_Outer_OOF_Predictions.csv`, `Candidate_Outer_OOF_Metrics.csv` | Eight algorithms' separate outer OOF predictions and metrics for comparison plots. |
| `Outer_Algorithm_Selections.csv` | Identify the algorithm selected in each outer fold. |
| `final/Candidate_Inner_Scores.csv` | Compare mean three-fold RMSE for final algorithm selection. |
| `final/candidates/*/Inner_OOF_Predictions.csv` | Plot each algorithm's inner predictions at its selected parameters. These are tuning results. |
| `final/Training_Fit_Predictions.csv` | Plot fitted predictions on the final training set. |
| `final/Predictions.csv` | Plot predictions on the independent test set. |
| `Evaluation_Metrics.csv` | Report R², RMSE, MAE, record count and molecule count for outer OOF, test and training-fit predictions. |
| `final/Model.joblib`, `final/Selector.json` | Store the fitted model, feature order, solvent vocabulary and feature settings for prediction. |

Each selection round also saves Optuna trials, best parameters and fold-specific
selectors. `Study_Config.json`, `Environment.json`, `Source_Hashes.json` and
`Artifact_Hashes.json` record the settings, dependency versions and file hashes.
Use a new output directory for each run. RMSE used for selection is the arithmetic
mean of the three fold RMSEs; pooled OOF RMSE is computed over all pooled records.

### 3. Predict

Prepare a CSV file with `SMILES` and `Solvent` columns, then run:

```bash
python predict.py --model outputs/paper/final \
  --input new_molecules.csv --output new_predictions.csv
```

The output preserves the input columns and adds `prediction_nm` (predicted
emission wavelength in nm) and `unknown_solvent` (whether the solvent was
absent from the model's training vocabulary). The saved selector supplies
the feature order and fingerprint settings. Use a new output filename for
each prediction run.

### 4. Check the training logic

```bash
python -m unittest discover -s tests -v
```
