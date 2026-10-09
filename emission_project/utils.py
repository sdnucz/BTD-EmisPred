"""Molecular, solvent and metric utilities."""
from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import Crippen,Descriptors,Fragments,Lipinski,MACCSkeys,rdFingerprintGenerator,rdMolDescriptors
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
FEATURE_PREFIXES = ("Morgan_", "MACCS_", "RDKit_", "Frag_", "Solvent_")

MACCS_KEY_COUNT = 166

SOLVENT_ALIASES = {
    "": "",
    "NAN": "",
    "NONE": "",
    "NA": "",
    "ACETONITRILE": "ACN",
    "MECN": "ACN",
    "CH3CN": "ACN",
    "CHLOROFORM": "CFM",
    "CHCL3": "CFM",
    "DICHLOROMETHANE": "DCM",
    "CH2CL2": "DCM",
    "METHYLENECHLORIDE": "DCM",
    "TOLUENE": "TOL",
    "WATER": "H2O",
    "ETHANOL": "ETOH",
    "METHANOL": "MEOH",
    "DIMETHYLFORMAMIDE": "DMF",
    "DIMETHYLSULFOXIDE": "DMSO",
    "TETRAHYDROFURAN": "THF",
    "ETHYLACETATE": "ACOET",
    "ETOAC": "ACOET",
    "DIOXANE": "DIOX",
    "PROPOH": "IPROPOH",
    "SOLIDSTATE": "SOLID",
}

RDKIT_DESCRIPTOR_FUNCTIONS = {
    "RDKit_MolWt": Descriptors.MolWt,
    "RDKit_MolLogP": Crippen.MolLogP,
    "RDKit_MolMR": Crippen.MolMR,
    "RDKit_TPSA": rdMolDescriptors.CalcTPSA,
    "RDKit_NumHDonors": Lipinski.NumHDonors,
    "RDKit_NumHAcceptors": Lipinski.NumHAcceptors,
    "RDKit_NumRotatableBonds": Lipinski.NumRotatableBonds,
    "RDKit_HeavyAtomCount": Lipinski.HeavyAtomCount,
    "RDKit_RingCount": Lipinski.RingCount,
    "RDKit_NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "RDKit_NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
    "RDKit_NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,
    "RDKit_FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "RDKit_NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
}

FRAGMENT_COUNT_FUNCTIONS = {
    "Frag_ArN": Fragments.fr_ArN,
    "Frag_Ar_N": Fragments.fr_Ar_N,
    "Frag_NH0": Fragments.fr_NH0,
    "Frag_NH1": Fragments.fr_NH1,
    "Frag_NH2": Fragments.fr_NH2,
    "Frag_aniline": Fragments.fr_aniline,
    "Frag_azide": Fragments.fr_azide,
    "Frag_azo": Fragments.fr_azo,
    "Frag_ether": Fragments.fr_ether,
    "Frag_nitrile": Fragments.fr_nitrile,
    "Frag_halogen": Fragments.fr_halogen,
    "Frag_C_S": Fragments.fr_C_S,
    "Frag_sulfide": Fragments.fr_sulfide,
    "Frag_sulfone": Fragments.fr_sulfone,
    "Frag_pyridine": Fragments.fr_pyridine,
}

@lru_cache(maxsize=None)
def get_morgan_generator(radius: int, n_bits: int) -> rdFingerprintGenerator.FingerprintGenerator64:
    """
    Return a cached RDKit Morgan fingerprint generator for the requested radius and bit length.
    """
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns by known feature prefixes."""
    return [column for column in df.columns if column.startswith(FEATURE_PREFIXES)]

def target_columns(df: pd.DataFrame) -> list[str]:
    """Return non-feature columns carried alongside model matrices."""
    return [column for column in df.columns if not column.startswith(FEATURE_PREFIXES)]

def normalize_solvent_label(value: Any) -> str:
    """Standardize solvent labels to compact uppercase aliases used for one-hot encoding."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", "", text)
    key = text.upper().replace("-", "").replace("_", "")
    return SOLVENT_ALIASES.get(key, key)

def solvent_to_feature_name(solvent: Any) -> str:
    """Convert a normalized solvent label into a Solvent_* feature column name."""
    solvent_label = normalize_solvent_label(solvent)
    token = re.sub(r"[^A-Za-z0-9]+", "_", solvent_label).strip("_")
    return f"Solvent_{token or 'UNKNOWN'}"

def build_solvent_feature_frame(
    solvent_values: pd.Series,
    solvent_categories: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Create a one-hot encoded solvent feature dataframe from raw solvent labels."""
    normalized = solvent_values.map(normalize_solvent_label)
    categories = sorted(normalized.dropna().astype(str).unique()) if solvent_categories is None else list(solvent_categories)
    feature_names = sorted(set(solvent_to_feature_name(category) for category in categories))
    feature_df = pd.DataFrame(0.0, index=solvent_values.index, columns=feature_names)
    name_by_category = {category: solvent_to_feature_name(category) for category in categories}
    for row_index, category in normalized.items():
        feature_name = name_by_category.get(str(category))
        if feature_name is not None:
            feature_df.loc[row_index, feature_name] = 1.0
    return feature_df.reset_index(drop=True)

def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation while returning NaN for degenerate inputs."""
    if len(y_true) < 2:
        return 0.0
    if np.allclose(np.std(y_true), 0) or np.allclose(np.std(y_pred), 0):
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute r, R2, RMSE and MAE for regression predictions."""
    return {
        "r": safe_pearsonr(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }

def robust_read_csv(file_path: Path) -> pd.DataFrame:
    """Read a CSV file using common UTF encodings and fall back to GBK when needed."""
    encodings = ["utf-8-sig", "gbk", "utf-8"]
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(file_path)

def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Create parent directories and save a dataframe as UTF-8 CSV."""
    df.to_csv(path, index=False, encoding="utf-8")

@lru_cache(maxsize=None)
def smiles_to_mol(smiles: Any) -> Chem.Mol | None:
    """Parse a SMILES string into an RDKit molecule or return None for invalid input."""
    smiles_text = str(smiles).strip()
    if not smiles_text:
        return None
    return Chem.MolFromSmiles(smiles_text)

@lru_cache(maxsize=None)
def canonicalize_smiles(smiles: Any) -> str | None:
    """Return canonical SMILES for valid molecules."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def smiles_to_morgan(smiles: Any, radius: int, n_bits: int) -> np.ndarray:
    """Convert a SMILES string into a binary Morgan fingerprint vector."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.int8)
    generator = get_morgan_generator(radius, n_bits)
    fingerprint = generator.GetFingerprint(mol)
    fingerprint_array = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    return fingerprint_array

def get_rdkit_descriptor_names() -> list[str]:
    """Return the configured RDKit descriptor feature names."""
    return list(RDKIT_DESCRIPTOR_FUNCTIONS)

def get_fragment_feature_names() -> list[str]:
    """Return the configured RDKit fragment-count feature names."""
    return list(FRAGMENT_COUNT_FUNCTIONS)

def get_maccs_key_names() -> list[str]:
    """Return MACCS key feature names."""
    return [f"MACCS_{index}" for index in range(1, MACCS_KEY_COUNT + 1)]

def get_maccs_key_smarts(key_index: int) -> str | None:
    """Return an example SMARTS pattern for a MACCS key index when available."""
    pattern = MACCSkeys.smartsPatts.get(int(key_index))
    if pattern is None:
        return None
    smarts = str(pattern[0])
    return None if smarts == "?" else smarts

@lru_cache(maxsize=None)
def get_feature_column_names(
    morgan_bits: int,
    use_morgan_features: bool,
    use_maccs_keys: bool,
    use_rdkit_descriptors: bool,
    use_fragment_features: bool,
) -> tuple[str, ...]:
    """Return feature column names in the exact order produced by smiles_to_feature_vector."""
    column_names: list[str] = []
    if use_morgan_features:
        column_names.extend(f"Morgan_{index}" for index in range(morgan_bits))
    if use_maccs_keys:
        column_names.extend(get_maccs_key_names())
    if use_rdkit_descriptors:
        column_names.extend(get_rdkit_descriptor_names())
    if use_fragment_features:
        column_names.extend(get_fragment_feature_names())
    if not column_names:
        raise ValueError("At least one feature block must be enabled.")
    return tuple(column_names)

def compute_rdkit_descriptor_vector(mol: Chem.Mol | None) -> np.ndarray:
    """Compute numeric RDKit descriptor values for one molecule."""
    if mol is None:
        return np.zeros(len(RDKIT_DESCRIPTOR_FUNCTIONS), dtype=np.float32)
    return np.asarray([float(func(mol)) for func in RDKIT_DESCRIPTOR_FUNCTIONS.values()], dtype=np.float32)

def compute_maccs_key_vector(mol: Chem.Mol | None) -> np.ndarray:
    """Compute binary MACCS key values for one molecule."""
    if mol is None:
        return np.zeros(MACCS_KEY_COUNT, dtype=np.float32)
    fingerprint = MACCSkeys.GenMACCSKeys(mol)
    return np.asarray([float(fingerprint.GetBit(index)) for index in range(1, MACCS_KEY_COUNT + 1)], dtype=np.float32)

def compute_fragment_count_vector(mol: Chem.Mol | None) -> np.ndarray:
    """Compute RDKit fragment-count values for one molecule."""
    if mol is None:
        return np.zeros(len(FRAGMENT_COUNT_FUNCTIONS), dtype=np.float32)
    return np.asarray([float(func(mol)) for func in FRAGMENT_COUNT_FUNCTIONS.values()], dtype=np.float32)

def smiles_to_feature_vector(
    smiles: Any,
    radius: int,
    n_bits: int,
    use_morgan_features: bool,
    use_maccs_keys: bool,
    use_rdkit_descriptors: bool,
    use_fragment_features: bool,
) -> tuple[np.ndarray, bool]:
    """
    Convert one SMILES string into the configured molecular feature vector.

    The vector can include Morgan fingerprints, MACCS keys, RDKit descriptors and
    RDKit fragment counts. Solvent one-hot features are appended later because they
    are row/table-level values rather than molecule-only descriptors.
    """
    mol = smiles_to_mol(smiles)
    feature_blocks: list[np.ndarray] = []

    if use_morgan_features:
        if mol is None:
            feature_blocks.append(np.zeros(n_bits, dtype=np.float32))
        else:
            feature_blocks.append(smiles_to_morgan(smiles, radius, n_bits).astype(np.float32, copy=False))
    if use_maccs_keys:
        feature_blocks.append(compute_maccs_key_vector(mol))
    if use_rdkit_descriptors:
        feature_blocks.append(compute_rdkit_descriptor_vector(mol))
    if use_fragment_features:
        feature_blocks.append(compute_fragment_count_vector(mol))

    if not feature_blocks:
        raise ValueError("At least one feature block must be enabled.")
    return np.concatenate(feature_blocks).astype(np.float32, copy=False), mol is not None
