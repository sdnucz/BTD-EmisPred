# NIR-II-EmisPred

This repository provides the code, exported model files, and lightweight web
application accompanying our NIR-II fluorescence emission wavelength prediction
study.

Given a molecular SMILES string and a solvent label, the packaged model predicts
the emission wavelength in nm. The current release is designed for model
deployment and web demonstration rather than full training-data redistribution.

## Overview

NIR-II-EmisPred combines molecular structure and solvent information to predict
near-infrared-II fluorescence emission wavelength.

The inference pipeline uses:

- Morgan fingerprints for local molecular topology.
- RDKit fragment-count features for interpretable substructure information.
- Solvent one-hot encoding for experimental solvent conditions.
- A fixed selected-feature table exported from the mainline training workflow.
- An XGBoost regression model as the base predictor.
- A gated nearest-neighbor residual correction module built from training OOF
  residuals.

The web app follows the same feature order and model artifacts as the command
line/package inference code.

## Repository Contents

```text
.
|-- README.md
|-- requirements.txt
|-- config.predict.mainline.yaml
|-- start_web_app.sh
|-- data/
|   |-- dataset.py
|   `-- prediction/
|       `-- SMILES_solvent_template.csv
|-- emission_project/
|   |-- infer.py
|   |-- model.py
|   `-- utils.py
|-- models/
|   `-- mainline/
|       |-- XGB_Final_Model.json
|       |-- Final_Model_Selected_Features.csv
|       |-- NN_Residual_Correction_Config.json
|       |-- NN_Residual_Correction_Library.csv
|       |-- Model_Artifacts_Metadata.json
|       `-- mainline_xgb_optuna_summary.json
`-- web_app/
    |-- app.py
    `-- assets/
```

## Installation

RDKit is usually most reliable through conda:

```bash
conda create -n nir2-emispred python=3.11 -y
conda activate nir2-emispred
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

On platforms where the PyPI RDKit build works, the following may also be
sufficient:

```bash
pip install -r requirements.txt
```

## Path Configuration

The repository uses relative paths by default through
`config.predict.mainline.yaml`. If you move the model files, deploy the web app
outside this repository, or adapt the metadata file, replace example paths such
as:

```text
/your/path/to/NIR-II-EmisPred-Paper
```

with the actual path on your own machine or server.

## Model Assets

The exported model files are stored in `models/mainline/`.

Key files:

- `XGB_Final_Model.json`: trained XGBoost model.
- `Final_Model_Selected_Features.csv`: fixed feature names and order used at
  inference time.
- `NN_Residual_Correction_Config.json`: nearest-neighbor correction settings.
- `NN_Residual_Correction_Library.csv`: OOF residual library used by the
  correction module.
- `Model_Artifacts_Metadata.json`: model, fingerprint, feature-block, and
  training-artifact metadata.

The fixed feature list is required. New molecules are featurized first and then
reindexed to this saved feature order before prediction.

## Quick Start

Run a self-test prediction:

```bash
python web_app/app.py \
  --self-test \
  --smiles "c1cc(Nc2ccncc2)c2nsnc2c1" \
  --solvent THF
```

Expected output fields include:

- raw XGBoost prediction
- nearest-neighbor correction
- final emission wavelength prediction
- maximum neighbor similarity
- correction status

## Web Application

Start the local web app:

```bash
python web_app/app.py --host 0.0.0.0 --port 7860
```

Then open:

```text
http://127.0.0.1:7860/home/prediction.html
```

The web interface accepts:

- `SMILES`
- `Solvent`

and returns:

- base XGBoost prediction
- nearest-neighbor residual correction
- final predicted emission wavelength in nm
- model confidence-related neighbor information

## Prediction Input Format

For reusable prediction workflows, follow the template:

```text
data/prediction/SMILES_solvent_template.csv
```

Required columns:

- `SMILES`
- `Solvent`

Solvent names should be standardized when possible, for example `THF`, `H2O`,
`Toluene`, or `DMSO`.

## Citation

If you use this repository, please cite the associated manuscript. Citation
details will be updated after publication.

## License

No open-source license has been specified yet. Please contact the authors before
redistributing the model artifacts or using them in a commercial setting.

## Contact

For questions about the model, web app, or paper companion files, please contact
the repository maintainers.
