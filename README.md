# NIR-II Emission Wavelength Predictor

This repository contains a packaged inference demo for a solvent-aware NIR-II molecular emission wavelength prediction model.

The package is intentionally limited to deployment and inference assets:

- exported XGBoost mainline model artifact
- fixed selected-feature list
- gated nearest-neighbor OOF residual correction library
- web prediction app
- reusable feature-generation and inference code

It does not include the original training dataset or historical experiment outputs.

## Input and Output

Input:

- `SMILES`: molecular structure string
- `Solvent`: solvent label, for example `THF`, `H2O`, `Toluene`, or `DMSO`

Output:

- raw XGBoost predicted emission wavelength
- nearest-neighbor residual correction
- final predicted emission wavelength in nm
- maximum nearest-neighbor similarity used by the correction gate

## Model Files

The model artifacts are in `models/mainline/`.

Key files:

- `XGB_Final_Model.json`
- `Final_Model_Selected_Features.csv`
- `NN_Residual_Correction_Config.json`
- `NN_Residual_Correction_Library.csv`
- `Model_Artifacts_Metadata.json`

## Metrics Snapshot

```csv
model_name,protocol,r,R2,RMSE,MAE,mean_abs_error_nm,median_abs_error_nm,max_abs_error_nm,count_abs_error_le_50nm,fraction_abs_error_le_50nm,mean_relative_error_pct,median_relative_error_pct,max_relative_error_pct,count_relative_error_le_5pct,n
mainline_xgb_oof_raw,train_oof_raw,0.9549431851498449,0.9108542901994958,61.78174093098058,41.2380499653012,41.2380499653012,26.547332763671875,425.45458984375,594,0.7122302158273381,6.590795970356457,4.259369547528944,57.524432327384666,465,834
mainline_xgb_oof_nn_corrected,train_oof_nn_corrected_fixed_user_specified_fixed_gate_user_specified,0.957591435659074,0.9169263922592003,59.64052449760174,38.15437551452575,38.15437551452575,22.176934517668656,447.0328149780564,639,0.7661870503597122,5.739300702788407,3.8540172689783225,47.54936998324501,501,834
mainline_xgb_test_raw,heldout_test_raw,0.9791259219463605,0.9584712581781085,43.331630846211425,32.52292314845713,32.52292314845713,26.025390625,179.37741088867188,156,0.7839195979899497,5.188966975764686,4.382049666144031,26.77274789383162,113,199
mainline_xgb_test_nn_corrected,heldout_test_nn_corrected_fixed_user_specified_fixed_gate_user_specified,0.9802670323830028,0.9587020161031432,43.21107522928837,32.103736875233736,32.103736875233736,26.97613525390625,200.98201979832248,156,0.7839195979899497,5.131626010812053,4.17299593275627,29.997316387809324,113,199
```

## Repository Layout

```text
.
├── config.predict.mainline.yaml
├── data/
│   ├── dataset.py
│   └── prediction/SMILES_solvent_template.csv
├── emission_project/
│   ├── infer.py
│   ├── model.py
│   └── utils.py
├── models/mainline/
└── web_app/
    ├── app.py
    └── assets/
```

## Install

RDKit installation is usually most reliable through conda:

```bash
conda create -n nir2-emispred python=3.11 -y
conda activate nir2-emispred
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

If your platform supports the PyPI RDKit build, `pip install -r requirements.txt` may be sufficient.

## Run Web App

```bash
python web_app/app.py --host 0.0.0.0 --port 7860
```

Open:

```text
http://127.0.0.1:7860/home/prediction.html
```

## Self Test

```bash
python web_app/app.py --self-test --smiles "c1cc(Nc2ccncc2)c2nsnc2c1" --solvent THF
```

## Notes

- This is an inference package, not a full training reproduction package.
- The raw training dataset is intentionally excluded.
- If model artifacts are updated, keep `config.predict.mainline.yaml` and `models/mainline/Model_Artifacts_Metadata.json` consistent with the exported files.
