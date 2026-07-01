from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from data.dataset import PathConfig, PipelineConfig, get_optional_solvent_col
from .model import get_model_display_name, load_final_model, resolve_explainability_model
from .utils import (
    build_solvent_feature_frame,
    get_feature_column_names,
    normalize_shap_values,
    plot_sample_shap_contributions,
    robust_read_csv,
    smiles_to_feature_vector,
)


def normalize_smiles_input_columns(df: pd.DataFrame) -> dict[str, str]:
    normalized_map = {
        str(column).strip().replace("\ufeff", "").lower(): column for column in df.columns
    }
    mapping: dict[str, str] = {}

    for logical_name in ["xuhao", "jiegou", "smiles", "solvent"]:
        if logical_name in normalized_map:
            mapping[logical_name] = normalized_map[logical_name]

    if "smiles" not in mapping:
        for column in df.columns:
            if "smiles" in str(column).lower():
                mapping["smiles"] = column
                break

    if "smiles" not in mapping:
        fallback_index = min(2, len(df.columns) - 1)
        mapping["smiles"] = df.columns[fallback_index]

    if "solvent" not in mapping:
        for column in df.columns:
            column_text = str(column).strip().lower()
            if "solvent" in column_text or "溶剂" in column_text:
                mapping["solvent"] = column
                break

    if "xuhao" not in mapping and len(df.columns) >= 1:
        mapping["xuhao"] = df.columns[0]
    if "jiegou" not in mapping and len(df.columns) >= 2:
        mapping["jiegou"] = df.columns[1]
    return mapping


def build_prediction_features(
    smiles_df: pd.DataFrame,
    mapping: dict[str, str],
    config: PipelineConfig,
    selected_features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    invalid_smiles: list[tuple[int, str]] = []
    feature_rows: list[np.ndarray] = []

    for index, smiles in enumerate(smiles_df[mapping["smiles"]]):
        feature_vector, is_valid = smiles_to_feature_vector(
            smiles,
            config.morgan_radius,
            config.morgan_bits,
            config.use_morgan_features,
            config.use_maccs_keys,
            config.use_rdkit_descriptors,
            config.use_fragment_features,
        )
        if not is_valid:
            invalid_smiles.append((index + 1, str(smiles)))
        feature_rows.append(feature_vector)

    feature_df = pd.DataFrame(
        feature_rows,
        columns=get_feature_column_names(
            config.morgan_bits,
            config.use_morgan_features,
            config.use_maccs_keys,
            config.use_rdkit_descriptors,
            config.use_fragment_features,
        ),
    )

    solvent_col = get_optional_solvent_col(config)
    if solvent_col is not None:
        solvent_input_col = mapping.get("solvent")
        if solvent_input_col is None:
            configured_key = solvent_col.strip().lower()
            normalized_columns = {str(column).strip().lower(): column for column in smiles_df.columns}
            solvent_input_col = normalized_columns.get(configured_key)
        if solvent_input_col is None:
            raise ValueError(
                "Prediction data must include a solvent column when pipeline.use_solvent_features is true. "
                f"Expected column like '{solvent_col}' or 'Solvent'."
            )
        solvent_feature_df = build_solvent_feature_frame(smiles_df[solvent_input_col])
        feature_df = pd.concat([feature_df, solvent_feature_df], axis=1)

    if selected_features is not None:
        for feature_name in selected_features:
            if feature_name not in feature_df.columns and str(feature_name).startswith("Solvent_"):
                feature_df[feature_name] = 0.0
    return feature_df, invalid_smiles


def load_selected_features(selected_features_path: Path) -> list[str]:
    if not selected_features_path.exists():
        raise FileNotFoundError(f"Selected feature file not found: {selected_features_path}")

    feature_df = robust_read_csv(selected_features_path)
    if feature_df.empty:
        raise ValueError(f"Selected feature file is empty: {selected_features_path}")

    column_name = "Selected_Features" if "Selected_Features" in feature_df.columns else feature_df.columns[0]
    selected_features = feature_df[column_name].dropna().astype(str).tolist()
    if not selected_features:
        raise ValueError(f"Selected feature file contains no usable features: {selected_features_path}")
    return selected_features


def validate_prediction_artifacts(paths: PathConfig, config: PipelineConfig, selected_features: list[str]) -> None:
    metadata_path = paths.artifact_metadata
    configured_feature_names = set(
        get_feature_column_names(
            config.morgan_bits,
            config.use_morgan_features,
            config.use_maccs_keys,
            config.use_rdkit_descriptors,
            config.use_fragment_features,
        )
    )
    missing_selected_features = [
        feature_name
        for feature_name in selected_features
        if feature_name not in configured_feature_names
        and not (config.use_solvent_features and str(feature_name).startswith("Solvent_"))
    ]
    if missing_selected_features:
        preview = ", ".join(missing_selected_features[:5])
        raise ValueError(
            "Prediction config cannot reproduce the saved training feature space. "
            f"Missing {len(missing_selected_features)} selected feature(s) under the current feature-block settings. "
            f"Examples: {preview}"
        )

    if not metadata_path.exists():
        return

    artifact_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_metadata = artifact_metadata.get("model", {})
    fingerprint_metadata = artifact_metadata.get("fingerprint", {})
    mismatches: list[str] = []

    trained_model_type = model_metadata.get("type")
    current_model_type = str(getattr(config, "final_model_type", "xgb")).strip().lower()
    if current_model_type == "ensemble_mean":
        current_model_type = "mean_ensemble"
    if trained_model_type not in {None, current_model_type}:
        mismatches.append(
            f"final_model_type: trained={trained_model_type}, current={current_model_type}"
        )

    if fingerprint_metadata.get("morgan_radius") not in {None, config.morgan_radius}:
        mismatches.append(
            f"morgan_radius: trained={fingerprint_metadata.get('morgan_radius')}, current={config.morgan_radius}"
        )
    if fingerprint_metadata.get("morgan_bits") not in {None, config.morgan_bits}:
        mismatches.append(
            f"morgan_bits: trained={fingerprint_metadata.get('morgan_bits')}, current={config.morgan_bits}"
        )

    selected_feature_count = artifact_metadata.get("selected_features", {}).get("count")
    if selected_feature_count not in {None, len(selected_features)}:
        mismatches.append(
            f"selected_feature_count: trained={selected_feature_count}, current={len(selected_features)}"
        )

    feature_block_metadata = artifact_metadata.get("feature_blocks", {})
    feature_block_pairs = {
        "use_morgan_features": config.use_morgan_features,
        "use_maccs_keys": config.use_maccs_keys,
        "use_rdkit_descriptors": config.use_rdkit_descriptors,
        "use_fragment_features": config.use_fragment_features,
        "use_solvent_features": config.use_solvent_features,
    }
    for key, current_value in feature_block_pairs.items():
        if feature_block_metadata.get(key) not in {None, current_value}:
            mismatches.append(
                f"{key}: trained={feature_block_metadata.get(key)}, current={current_value}"
            )

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Prediction config does not match the saved training artifacts. "
            f"Please align the config file or use the correct artifact set. Details: {mismatch_text}"
        )


def load_prediction_artifacts(paths: PathConfig, config: PipelineConfig) -> tuple[Any, list[str]]:
    artifact_metadata: dict[str, Any] = {}
    if paths.artifact_metadata.exists():
        artifact_metadata = json.loads(paths.artifact_metadata.read_text(encoding="utf-8"))

    trained_model_type = artifact_metadata.get("model", {}).get("type")
    if trained_model_type not in {None, "xgb", "mean_ensemble"}:
        raise ValueError(
            "Only XGB and mean-ensemble artifacts are supported now. "
            f"Found saved model type: {trained_model_type}"
        )

    model = load_final_model(paths.trained_model)
    selected_features = load_selected_features(paths.selected_features_data)
    validate_prediction_artifacts(paths, config, selected_features)
    return model, selected_features


def run_prediction_workflow(
    model: Any,
    selected_features: list[str],
    paths: PathConfig,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, str]]]:
    smiles_df = robust_read_csv(paths.prediction_data)
    mapping = normalize_smiles_input_columns(smiles_df)
    feature_df, invalid_smiles = build_prediction_features(smiles_df, mapping, config, selected_features)
    x_pred = feature_df[selected_features].values
    predictions = model.predict(x_pred)

    result_data = {
        "序号": smiles_df[mapping["xuhao"]] if "xuhao" in mapping else np.arange(1, len(smiles_df) + 1),
        "SMILES": smiles_df[mapping["smiles"]],
        "预测发射波长_em (nm)": predictions,
    }
    if "jiegou" in mapping:
        result_data["结构类型"] = smiles_df[mapping["jiegou"]]
    if "solvent" in mapping:
        result_data["溶剂"] = smiles_df[mapping["solvent"]]

    result_df = pd.DataFrame(result_data)
    result_df.to_csv(paths.output_dir / "SMILES_预测结果_XGB.csv", index=False, encoding="utf-8-sig")
    result_df.to_csv(paths.output_dir / "SMILES_预测结果_Mainline.csv", index=False, encoding="utf-8-sig")

    if invalid_smiles:
        invalid_df = pd.DataFrame(invalid_smiles, columns=["样本序号", "无效SMILES"])
        invalid_df.to_csv(paths.output_dir / "Invalid_SMILES_Report.csv", index=False, encoding="utf-8-sig")

    return smiles_df, feature_df, invalid_smiles


def run_prediction_shap_analysis(
    model: Any,
    smiles_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    selected_features: list[str],
    paths: PathConfig,
    config: PipelineConfig,
) -> None:
    explainability_model, explainability_mode = resolve_explainability_model(model)
    if explainability_model is None:
        (paths.output_dir / "Prediction_Explainability_Notice.txt").write_text(
            f"Explainability is not available for final model '{get_model_display_name(model)}'.",
            encoding="utf-8",
        )
        return

    mapping = normalize_smiles_input_columns(smiles_df)
    x_pred = feature_df[selected_features].values
    sample_count = min(config.prediction_shap_limit, x_pred.shape[0])
    if sample_count == 0:
        return

    explainer = shap.TreeExplainer(explainability_model)
    shap_values = normalize_shap_values(explainer.shap_values(x_pred[:sample_count]))
    predictions = model.predict(x_pred[:sample_count])
    summary_rows: list[dict[str, object]] = []

    for row_index in range(sample_count):
        shap_row = shap_values[row_index]
        positive_index = int(np.argmax(shap_row))
        negative_index = int(np.argmin(shap_row))
        smiles_text = str(smiles_df.iloc[row_index][mapping["smiles"]])

        summary_rows.append(
            {
                "样本序号": row_index + 1,
                "SMILES": smiles_text,
                "预测em值(nm)": round(float(predictions[row_index]), 2),
                "正向关键特征": selected_features[positive_index],
                "正向特征SHAP值": round(float(shap_row[positive_index]), 4),
                "负向关键特征": selected_features[negative_index],
                "负向特征SHAP值": round(float(shap_row[negative_index]), 4),
                "解释来源": explainability_mode,
            }
        )

        plot_sample_shap_contributions(
            selected_features,
            shap_row,
            row_index + 1,
            paths.output_dir / f"SMILES_Sample_{row_index + 1}_SHAP.png",
            config.sample_plot_feature_limit,
        )

    pd.DataFrame(summary_rows).to_csv(
        paths.output_dir / "SMILES_Single_Sample_SHAP_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
