# BTD-EmisPred

BTD-EmisPred predicts fluorescence emission wavelengths of BTD-based molecules
from molecular SMILES and solvent information. It combines Morgan fingerprints,
RDKit fragment counts and solvent one-hot features with regression models.
Our trained model is available on the [BTD-EmisPred website](https://btd-emispred.online).

## Files

| File | Purpose |
| --- | --- |
| `reproduce.py` | Refit the archived candidate parameters or repeat the parameter searches using the fixed partitions. |
| `compare_reproduction.py` | Compare newly fitted selectors, algorithm choices, scores and predictions with the reference results. |
| `train.py` | Run molecule-grouped nested cross-validation, candidate selection, final fitting and test evaluation. |
| `predict.py` | Predict emission wavelengths from a trained model and its selector. |
| `plot_review_results.py` | Generate outer-CV comparison figures and long-wavelength evaluation tables from run outputs. |
| `data/dataset.py` | Clean records, group molecules, construct features and fit fold-local feature selection. |
| `emission_project/utils.py` | Molecular, solvent and evaluation utilities. |
| `data/data/nir2_emission_dataset.csv` | The 1,012-record dataset, with 28 solvent/environment categories. |
| `configs/paper.json` | Feature settings, parameter search spaces, fixed GBR parameters, random seed and thread settings. |
| `configs/Estimator_Parameters.json` | Candidate parameters and reference scores for ten outer rounds and final selection. |
| `configs/Selector.json` | The final model's ordered 82-feature selector and preprocessing information. |
| `configs/Environment.json` | Software versions used for the reference results. |
| `configs/Final_Selection.json` | Reference final algorithm selection. |
| `frozen_partitions.py` | Validate the dataset and load molecule-grouped partitions. |
| `splits/Frozen_Partitions.json` | Training/test, outer-CV and inner-CV record IDs. |
| `splits/Dataset_SHA256.txt` | Dataset checksum. |
| `reference/` | Expected predictions and selector check values, used only after fitting. |
| `requirements.txt` | Pinned Python dependencies. |
| `tests/test_reproduction.py` | Targeted checks for fixed parameters, feature order and reproduction comparisons. |

## Install

Use Python 3.12:

```bash
conda create -n your_env python=3.12 -y
conda activate your_env
pip install -r requirements.txt
```

## Reproduce the archived results

```bash
python reproduce.py --mode refit --output outputs/reproduced
python compare_reproduction.py --run outputs/reproduced
```

This refits preprocessing, feature selection and all eight candidates from the
supplied data in each training partition, using the archived candidate parameters.
It recalculates the three inner-fold scores, selects the lowest mean RMSE and
refits the selected model. It does not load the archived model or reference
predictions for fitting. Reference predictions are read only by the comparison
script. A successful comparison writes `Reproduction_Verification.json`.

The fixed split contains 816 training records (366 molecules) and 196 test records
(94 molecules). IDs refer to cleaned zero-based `record_id` values. The ten outer
rounds evaluate the selection procedure; a separate three-fold selection on the
full training set yields the final LightGBM model.

## Repeat the parameter searches

```bash
python reproduce.py --mode search --output outputs/search
python compare_reproduction.py --run outputs/search
```

The archived result compares eight candidate algorithms: XGBoost, CatBoost,
LightGBM, RF, GBR, KNN, KRR and SVR. Seven algorithms undergo 50 Optuna trials per
selection round. GBR uses parameters inherited from the prior 1,011-record final
training selection, evaluated on the same three inner folds without a new GBR
search. This gives 3,850 search attempts across eleven selection rounds, plus
fixed-parameter GBR evaluations. It is a reconstruction of that historical
comparison, rather than eight newly tuned algorithms. The inherited GBR comparison
is conditional on that earlier parameter selection.

| Setting | Value |
| --- | --- |
| Outer / inner molecule-grouped CV | 10 / 3 folds |
| Morgan fingerprint | Radius 2, 2,048 bits |
| RDKit fragments | 15 |
| Absolute correlation threshold | 0.5 |
| RF-RFE retained features | 82, including protected Morgan_1073 |
| RF-RFE trees / maximum depth / elimination step | 150 / 10 / 20 |
| Random seed / stratification bins | 2 / 5 |
| Estimator threads / concurrent outer folds | 2 / 1 |

Feature selection is fitted independently inside each inner training fold and
again on the corresponding full outer training set. The final selector is fitted
on all training records. `solvent_feature_order` preserves the numerical column
order of the archived fit after the MTBE label update; it does not fit a vocabulary
on validation data. Each training fold derives its own available solvent categories.
The original two-thread setting is retained because KNN predictions with tied
neighbor distances can differ when threading changes.

`refit` verifies the fitted result conditional on saved parameters. `search`
additionally repeats Optuna optimization. Use the pinned environment; different
numerical libraries, hardware or thread settings can affect tie handling and
optimization trajectories. The comparison script checks 44 selectors, 88 candidate
parameter/score sets, procedure OOF, eight-algorithm outer OOF, final inner predictions,
training/test predictions and external examples. Prediction tolerance is 1e-7 nm.

## Predict new molecules

Prepare a CSV with `SMILES` and `Solvent` columns:

```bash
python predict.py --model outputs/reproduced/final \
  --input new_molecules.csv --output new_predictions.csv
```

The output adds `prediction_nm` and `unknown_solvent`. The matching saved selector
supplies the feature order and solvent vocabulary. Use a new output directory or
filename for every run.

## Generate evaluation figures

```bash
python plot_review_results.py --run outputs/reproduced --output outputs/figures
```

This exports PNG, PDF, SVG and CSV files for eight-algorithm outer-CV comparisons,
outer selections and evaluation at experimental wavelengths of at least 1,000 nm.
Scatter-plot metrics use pooled predictions; comparison bars use the mean and
sample SD across ten folds. Use `--font-dir` to supply local fonts.

## Download the model and figure-source archive

The matching model, full selector states, search records, predictions and source
figure data are provided in the
[final LightGBM reproduction release](https://github.com/sdnucz/BTD-EmisPred/releases/tag/v1.3.2-final-lightgbm-reproduction).
