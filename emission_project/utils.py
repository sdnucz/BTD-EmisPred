from __future__ import annotations

import re
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Fragments, Lipinski, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith(FEATURE_PREFIXES)]


def target_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if not column.startswith(FEATURE_PREFIXES)]


def normalize_solvent_label(value: Any) -> str:
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
    solvent_label = normalize_solvent_label(solvent)
    token = re.sub(r"[^A-Za-z0-9]+", "_", solvent_label).strip("_")
    return f"Solvent_{token or 'UNKNOWN'}"


def build_solvent_feature_frame(
    solvent_values: pd.Series,
    solvent_categories: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
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
    if len(y_true) < 2:
        return 0.0
    if np.allclose(np.std(y_true), 0) or np.allclose(np.std(y_pred), 0):
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r": safe_pearsonr(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def robust_read_csv(file_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "gbk", "utf-8"]
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(file_path)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


@lru_cache(maxsize=None)
def smiles_to_mol(smiles: Any) -> Chem.Mol | None:
    smiles_text = str(smiles).strip()
    if not smiles_text:
        return None
    return Chem.MolFromSmiles(smiles_text)


@lru_cache(maxsize=None)
def canonicalize_smiles(smiles: Any) -> str | None:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_morgan(smiles: Any, radius: int, n_bits: int) -> np.ndarray:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.int8)
    generator = get_morgan_generator(radius, n_bits)
    fingerprint = generator.GetFingerprint(mol)
    fingerprint_array = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    return fingerprint_array


def get_rdkit_descriptor_names() -> list[str]:
    return list(RDKIT_DESCRIPTOR_FUNCTIONS)


def get_fragment_feature_names() -> list[str]:
    return list(FRAGMENT_COUNT_FUNCTIONS)


def get_maccs_key_names() -> list[str]:
    return [f"MACCS_{index}" for index in range(1, MACCS_KEY_COUNT + 1)]


def get_maccs_key_smarts(key_index: int) -> str | None:
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
    if mol is None:
        return np.zeros(len(RDKIT_DESCRIPTOR_FUNCTIONS), dtype=np.float32)
    return np.asarray([float(func(mol)) for func in RDKIT_DESCRIPTOR_FUNCTIONS.values()], dtype=np.float32)


def compute_maccs_key_vector(mol: Chem.Mol | None) -> np.ndarray:
    if mol is None:
        return np.zeros(MACCS_KEY_COUNT, dtype=np.float32)
    fingerprint = MACCSkeys.GenMACCSKeys(mol)
    return np.asarray([float(fingerprint.GetBit(index)) for index in range(1, MACCS_KEY_COUNT + 1)], dtype=np.float32)


def compute_fragment_count_vector(mol: Chem.Mol | None) -> np.ndarray:
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


def normalize_shap_values(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    values = np.asarray(shap_values)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim == 3:
        values = values[0]
    return values


def configure_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_feature_similarity_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    configure_plot_style()
    features = feature_columns(df)
    corr_matrix = df[features].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(12, 10), dpi=600)
    heatmap = sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        annot=False,
        linewidths=0.1,
        cbar_kws={
            "shrink": 0.8,
            "label": "Pearson Correlation Coefficient",
            "ticks": [-1, -0.5, 0, 0.5, 1],
        },
    )

    ax.set_xlabel("Feature Index", fontsize=30, labelpad=12)
    ax.set_ylabel("Feature Index", fontsize=30, labelpad=12)

    tick_positions = list(range(0, len(features), 5))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_positions, rotation=0, fontsize=26, fontweight="medium")
    ax.set_yticklabels(tick_positions, rotation=0, fontsize=26, fontweight="medium")

    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=27)
    colorbar.set_label(
        "Pearson Correlation Coefficient",
        fontsize=30,
        fontweight="medium",
        labelpad=15,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_cv_regression_curves(
    summary_df: pd.DataFrame,
    fold_predictions: dict[str, dict[str, list[float]]],
    output_dir: Path,
) -> None:
    configure_plot_style()
    model_styles = {
        "RF": {"color": "#1f77b4", "marker": "o"},
        "KRR": {"color": "#ff7f0e", "marker": "o"},
        "XGB": {"color": "#2ca02c", "marker": "o"},
        "LGB": {"color": "#d62728", "marker": "o"},
        "SVR": {"color": "#9467bd", "marker": "o"},
        "ANN": {"color": "#8c564b", "marker": "o"},
    }

    for model_name, prediction_pack in fold_predictions.items():
        y_true = np.asarray(prediction_pack["y_true"])
        y_pred = np.asarray(prediction_pack["y_pred"])
        if y_true.size == 0:
            continue

        metrics_row = summary_df.loc[summary_df["Model"] == model_name].iloc[0]
        fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

        ax.scatter(
            y_true,
            y_pred,
            c=model_styles[model_name]["color"],
            marker=model_styles[model_name]["marker"],
            s=30,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            label="Predictions",
        )

        min_val = min(y_true.min(), y_pred.min()) - 10
        max_val = max(y_true.max(), y_pred.max()) + 10
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            linewidth=1.5,
            alpha=0.9,
            label="Ideal fit (y=x)",
        )

        metrics_text = (
            f"R² = {metrics_row['Mean R²']:.4f}\n"
            f"RMSE = {metrics_row['Mean RMSE (nm)']:.2f} nm\n"
            f"MAE = {metrics_row['Mean MAE (nm)']:.2f} nm"
        )
        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray", "alpha": 0.8},
            verticalalignment="top",
            fontsize=16,
        )

        ax.set_title(model_name, fontsize=30, pad=15)
        ax.set_xlabel("λ$_{em}$ Exp. (nm)", fontsize=30, labelpad=10)
        ax.set_ylabel("λ$_{em}$ Pred. (nm)", fontsize=30, labelpad=10)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.6)
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
        ax.tick_params(axis="both", which="major", labelsize=26, width=0.8)

        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_Regression_Curve.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def plot_test_regression_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, float],
    output_path: Path,
    title: str = "XGB Test Set",
) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    ax.scatter(
        y_true,
        y_pred,
        c="#2ca02c",
        marker="o",
        s=40,
        alpha=0.7,
        edgecolors="white",
        linewidth=0.6,
        label="Test set predictions",
    )

    min_val = min(y_true.min(), y_pred.min()) - 10
    max_val = max(y_true.max(), y_pred.max()) + 10
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5, alpha=0.9, label="Ideal fit (y=x)")

    metrics_text = (
        f"R² = {metrics['R2']:.4f}\n"
        f"RMSE = {metrics['RMSE']:.2f} nm\n"
        f"MAE = {metrics['MAE']:.2f} nm"
    )
    ax.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "lightgray", "alpha": 0.85},
        verticalalignment="top",
        fontsize=16,
    )

    ax.set_title(title, fontsize=30, pad=15)
    ax.set_xlabel("λ$_{em}$ Exp. (nm)", fontsize=30, labelpad=10)
    ax.set_ylabel("λ$_{em}$ Pred. (nm)", fontsize=30, labelpad=10)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.tick_params(axis="both", which="major", labelsize=26, width=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_substructure(smarts: str | None) -> np.ndarray | None:
    if smarts is None:
        return None

    if pd.isna(smarts):
        return None

    smarts_text = str(smarts).strip()
    if not smarts_text or smarts_text.lower() == "nan":
        return None

    mol = Chem.MolFromSmarts(smarts_text)
    if mol is None and smarts_text.isalpha() and not smarts_text.startswith("["):
        bracketed_text = f"[{smarts_text}]"
        mol = Chem.MolFromSmarts(bracketed_text)
        if mol is None:
            mol = Chem.MolFromSmiles(bracketed_text)
    if mol is None:
        return None

    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    drawer.SetFontSize(14)
    try:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    except Exception:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    image_bytes = BytesIO(drawer.GetDrawingText())
    try:
        return plt.imread(image_bytes, format="png")
    except Exception:
        return None


def plot_top_feature_substructures(top_features: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(10, 2.2 * len(top_features)), dpi=600)
    grid = gridspec.GridSpec(len(top_features) + 1, 3, width_ratios=[1, 2, 1])

    headers = ["Feature", "Substructure / Descriptor", "Effect on output"]
    for column_index, header in enumerate(headers):
        axis = fig.add_subplot(grid[0, column_index])
        axis.text(0.5, 0.5, header, ha="center", va="center", fontsize=16, fontweight="bold", color="darkblue")
        axis.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor="darkblue", linewidth=1.5))
        axis.axis("off")

    for row_index, (_, row) in enumerate(top_features.iterrows(), start=1):
        axis_feature = fig.add_subplot(grid[row_index, 0])
        axis_feature.text(0.5, 0.5, row["Feature_ID"], ha="center", va="center", fontsize=14, fontweight="medium")
        axis_feature.axis("off")

        axis_structure = fig.add_subplot(grid[row_index, 1])
        structure_image = draw_substructure(row["Substructure_SMARTS"])
        if structure_image is not None:
            axis_structure.imshow(structure_image)
        else:
            axis_structure.add_patch(
                plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, facecolor="lightgray", alpha=0.3)
            )
            fallback_text = str(row.get("Display_Label", "Substructure Not Found"))
            axis_structure.text(
                0.5,
                0.5,
                fallback_text,
                ha="center",
                va="center",
                fontsize=11,
                color="darkred",
                style="italic",
            )
        axis_structure.axis("off")

        axis_effect = fig.add_subplot(grid[row_index, 2])
        effect_color = "#2E8B57" if row["Effect"] == "Increase" else "#DC143C"
        axis_effect.text(0.5, 0.5, row["Effect"], ha="center", va="center", fontsize=14, fontweight="bold", color=effect_color)
        axis_effect.axis("off")

    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_sample_shap_contributions(
    feature_names: list[str],
    shap_row: np.ndarray,
    sample_index: int,
    output_path: Path,
    max_features: int,
) -> None:
    top_indices = np.argsort(np.abs(shap_row))[-max_features:]
    top_values = shap_row[top_indices]
    top_features = [feature_names[index] for index in top_indices]
    colors = ["#2E8B57" if value >= 0 else "#DC143C" for value in top_values]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=600)
    ax.barh(range(len(top_indices)), top_values, color=colors)
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels(top_features, fontsize=12)
    ax.set_xlabel("SHAP value", fontsize=16)
    ax.set_title(f"Sample {sample_index} SHAP Contributions", fontsize=18, pad=12)
    ax.axvline(0, color="black", linewidth=1)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
