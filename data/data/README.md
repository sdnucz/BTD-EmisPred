# Dataset

`nir2_emission_dataset.csv` is the cleaned molecule-solvent emission dataset used by the retraining workflow.

Columns:

- `canonical_smiles`: RDKit-canonicalized SMILES.
- `Solvent`: standardized solvent label used for one-hot encoding.
- `source_row`: row index in the cleaned source table.
- `Molecule`: molecule record index from the curated table.
- `SMILES`: original SMILES string used for feature generation.
- `λem (nm)`: experimental emission wavelength label in nm.
- `Solvent_raw`: original solvent label before standardization.
- `doi`: literature source DOI where available.

The default training and prediction configs point to this file. If you replace it with your own dataset, update `paths.raw_data_file`, `pipeline.target_col`, `pipeline.smiles_col`, and `pipeline.solvent_col` in the YAML files.

## License

This cleaned dataset is provided for academic reuse under CC BY 4.0. Please cite
the original literature sources listed in the `doi` column where appropriate.
