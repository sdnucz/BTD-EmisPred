# BTD-EmisPred

BTD-EmisPred predicts the fluorescence emission wavelength of BTD-based
molecules from molecular SMILES and solvent information. It combines molecular
fingerprints, RDKit fragment counts and solvent features with machine-learning
regression. Training uses molecule-grouped nested cross-validation for feature
selection, algorithm comparison and parameter optimization.

## Online prediction

Our trained model is deployed on the [BTD-EmisPred website](https://btd-emispred.online)
and can be used directly in your browser. Enter a molecular SMILES string and
select a solvent to obtain the predicted emission wavelength in nm.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Run nested cross-validation, select an algorithm and its parameters, fit the final model and evaluate test predictions. |
| `predict.py` | Predict emission wavelengths for new molecule–solvent records using a trained model. |
| `plot_review_results.py` | Recompute long-wavelength subset metrics and draw eight-algorithm outer-CV comparison figures from saved predictions. |
| `data/dataset.py` | Clean records, group molecules, construct features and perform feature selection within training folds. |
| `emission_project/utils.py` | Provide shared molecular, solvent, data-reading and evaluation utilities. |
| `data/data/nir2_emission_dataset.csv` | Store molecular structures, solvents, experimental emission wavelengths and source information. |
| `requirements.txt` | List the Python dependencies and their versions. |
| `configs/paper.json` | Configure data splitting, feature construction, feature selection and algorithm parameter searches. |
| `frozen_partitions.py` | Validate and load fixed molecule-grouped partitions. |
| `splits/Frozen_Partitions.json` | Record the training/test split, ten outer folds and three inner folds for each selection round. |
| `splits/Dataset_SHA256.txt` | Bind the fixed partitions to the supplied dataset. |
| `configs/Selector.json` | Reference full-training feature order, solvent vocabulary and RF-RFE preprocessing state. |

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

Solvent names are normalized consistently during training and prediction:
PHME → TOL, TCM → CFM, PBSBUFFER → PBS, and THF\\H2O → THF/H2O.
The supplied dataset contains 1,012 molecule–environment records across
28 solvent/environment categories. The `Solvent` column supplies the labels
used for feature encoding. MTBE denotes methyl tert-butyl ether.

Run training with the supplied configuration:

```bash
python train.py --data data/data/nir2_emission_dataset.csv \
  --config configs/paper.json --partitions splits/Frozen_Partitions.json \
  --output outputs/run
```

The fixed split contains 816 training records (366 molecules) and 196 test
records (94 molecules), with no molecule shared between them. Partition IDs are
zero-based `record_id` values assigned by `load_data` after cleaning and
deduplication, rather than source molecule numbers. The archive includes three
inner folds for each outer training set and for final selection on all training
records. The loader checks dataset integrity, fold coverage and molecular separation.

For a different dataset, omit `--partitions` to generate splits using the configuration.

The supplied configuration uses:

| Setting | Value |
| --- | --- |
| Molecule-grouped training/test split | Approximately 80% / 20% |
| Outer / inner cross-validation | 10 / 3 folds |
| Morgan fingerprint length | 2,048 bits |
| Morgan fingerprint radius | 2 |
| Absolute correlation threshold | 0.5 |
| Features retained by random-forest RFE | 82, including protected Morgan_1073 |
| RF-RFE trees / maximum depth / elimination step | 150 / 10 / 20 |
| Optuna trials per algorithm per selection round | 50 for each of the eight algorithms |
| Candidate algorithms | XGBoost, CatBoost, LightGBM, RF, GBR, KNN, KRR, SVR |
| Random seed | 2 |
| Molecular-target stratification bins | 5 |
| CPU threads / concurrent outer folds | 10 / 1 |

Within each outer training fold, preprocessing and feature selection are fitted
on the inner training folds. The lowest mean inner-fold RMSE selects the algorithm
and parameters. All eight algorithms undergo 50 Optuna trials each on the same
inner folds, including GBR. The selected pipeline
is refitted on the full outer training fold
to predict its outer test fold. After outer CV, the same inner selection procedure
is repeated on the full training set, followed by refitting and held-out testing.

Edit `configs/paper.json`, or copy it to a separate JSON file and pass that
file with `--config`. The configuration contains:

- Data splitting: `test_size`, `outer_folds`, `inner_folds`, `stratify_bins`.
- Feature construction: `morgan_radius`, `morgan_bits`.
- Feature selection: `selected_features`, `correlation_threshold`,
  `rfe_estimators`, `rfe_depth`, `rfe_step`, `protected_features`.
- Parameter search: `n_trials`, `search_spaces`.

`seed`, `threads` and `outer_jobs` control reproducibility and CPU use.
`search_spaces` maps candidate names (`XGB`, `CAT`, `LGBM`, `RF`, `GBR`,
`KNN`, `KRR`, `SVR`) to estimator parameters. Each parameter can be a fixed
value or a search specification: `kind` = `int` / `float` with `low`, `high`
and optional `step` / `log`, or `kind` = `categorical` with `choices`.

`protected_features` remain in the feature set throughout variance filtering,
correlation filtering and RF-RFE. They count toward the total of 82 features.
LightGBM participates in algorithm selection and uses the same saved-feature
prediction interface as the other estimators.

`configs/Selector.json` records the 82 features obtained on the fixed full
training set with `configs/paper.json`. It is a reference for checking the final
selector; training refits feature selection separately in every training fold.
Use the selector saved alongside each trained model for prediction.

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

### 4. Recompute comparison tables and figures

Keep `export_candidate_predictions` enabled when training. Then run:

```bash
python plot_review_results.py --run outputs/run --output outputs/review_figures
```

The script checks outer-fold coverage, molecule separation, inner-score selections
and agreement between selected-algorithm predictions and the procedure OOF predictions.
It exports MAE/RMSE and record/molecule counts for experimental wavelengths
at or above 1,000 nm, eight-algorithm scatter plots, fold-level comparison plots,
and each outer fold's selected algorithm. Scatter-plot metrics pool the OOF
predictions; comparison bars show the mean ± sample SD across ten folds, with
all ten fold values displayed. PDF, SVG, PNG and CSV files are written to the
specified output directory. Use `--font-dir` to load local TTF fonts; Arial is
used when available.
