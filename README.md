# BTD-EmisPred

BTD-EmisPred provides reusable training and prediction code for emission
wavelength prediction from molecular SMILES and solvent information.

This version contains four code files, the cleaned dataset and installation
information. Experiment settings are supplied at runtime. Dataset partitions, fitted models
and results are generated locally rather than bundled with the code.

## Files

- `train.py`: molecule-grouped nested validation, candidate selection and final fitting.
- `predict.py`: predictions using a saved model and selector.
- `data/dataset.py`: cleaning, grouping, molecular/solvent features and fold-local selection.
- `emission_project/utils.py`: shared molecular, solvent and metric utilities.
- `data/data/nir2_emission_dataset.csv`: cleaned molecule-solvent records with literature provenance.
- `requirements.txt`, `LICENSE` and `.gitignore`.

## Installation

The code was checked with Python 3.12 and the dependency versions below.

```bash
conda create -n nir2-emispred python=3.12 -y
conda activate nir2-emispred
pip install -r requirements.txt
```

## Training

Training data require `SMILES`, `Solvent` and `λem (nm)` columns.
Create your own local JSON configuration; no experiment configuration is shipped.

Required configuration keys are `test_size`, `outer_folds`, `inner_folds`,
`stratify_bins`, `selected_features`, `correlation_threshold`,
`rfe_estimators`, `rfe_depth`, `rfe_step`, `n_trials`,
`morgan_radius`, `morgan_bits` and `search_spaces`.
Optional `seed`, `threads` and `outer_jobs` control reproducibility and CPU use.

`search_spaces` maps candidate names (`XGB`, `CAT`, `LGBM`, `RF`, `GBR`,
`KNN`, `KRR`, `SVR`) to estimator parameter specifications. Each parameter
may be a fixed value, an `int` or `float` specification with `low`, `high`
and optional `step`/`log`, or a `categorical` specification with `choices`.
The specification type is given in `kind`.

```bash
python train.py --data data/data/nir2_emission_dataset.csv \
  --config settings.json --output outputs/run
```

Molecules are grouped before splitting. Solvent encoding, variance/correlation
filtering and RF-RFE are fitted within each training fold. KNN, KRR and SVR also
standardize selected inputs using training-fold statistics. Mean inner-fold RMSE
selects the candidate and parameters; alphabetical algorithm order resolves
exact ties. Outer OOF predictions evaluate the selection procedure. Selection is
repeated on all training records before final test evaluation. No residual
correction is applied.

Outputs include partitions, selectors, search records, models, predictions,
metrics and hashes under the requested output directory. These generated files
are excluded from Git. Existing output directories are rejected.

## Prediction

Prediction inputs require `SMILES` and `Solvent`.

```bash
python predict.py --model outputs/run/final \
  --input new_molecules.csv --output new_predictions.csv
```

The saved selector defines the feature order and fingerprint settings.
Outputs include `prediction_nm` and `unknown_solvent`; unseen solvents are
encoded as all zeros and flagged. Invalid structures or missing solvents are
rejected. For an older selector without fingerprint settings, supply your own
local feature configuration with `--config`.
Only load serialized model files produced by a trusted training run.

## License

The source code is released under the BSD 3-Clause License. The cleaned dataset
is provided for academic reuse under CC BY 4.0, with literature DOI provenance
retained in the `doi` column.
