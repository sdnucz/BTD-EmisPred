"""
Held-out test evaluation and explainability outputs.

This module evaluates the trained model on the fixed test partition, exports
metrics and error tables, and builds SHAP-based feature importance and
substructure mapping files for manuscript interpretation.
"""
from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from data.dataset import PathConfig, PipelineConfig, PreparedData, build_feature_selection_summary
from .model import MeanRegressorEnsemble, get_model_artifact_type, get_model_display_name, resolve_explainability_model
from .utils import (
    compute_metrics,
    configure_plot_style,
    get_maccs_key_smarts,
    get_morgan_generator,
    normalize_shap_values,
    plot_sample_shap_contributions,
    plot_test_regression_curve,
    plot_top_feature_substructures,
    save_dataframe,
)


def extract_model_hyperparameters(model: Any) -> dict[str, Any]:
    """Return a compact dictionary of trained model hyperparameters for run metadata."""
    if isinstance(model, MeanRegressorEnsemble):
        return {
            "ensemble_mode": "mean",
            "base_estimators": list(model.named_estimators_.keys()),
            "base_estimator_types": {
                name: type(estimator).__name__
                for name, estimator in model.named_estimators_.items()
            },
        }

    if hasattr(model, "named_estimators_") and hasattr(model, "final_estimator_"):
        return {
            "stacking_cv": getattr(model, "cv", None),
            "stacking_passthrough": getattr(model, "passthrough", None),
            "base_estimators": list(getattr(model, "named_estimators_", {}).keys()),
            "meta_estimator": type(getattr(model, "final_estimator_", None)).__name__,
        }

    if not hasattr(model, "get_params"):
        return {}

    model_params = model.get_params()
    relevant_keys = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    ]
    return {key: model_params.get(key) for key in relevant_keys}


def run_shap_analysis(
    model: Any,
    prepared: PreparedData,
    paths: PathConfig,
    config: PipelineConfig,
) -> pd.DataFrame:
    """
    Run TreeSHAP on a sample of the held-out test set and export feature-importance tables and plots.
    """
    explainability_model, explainability_mode = resolve_explainability_model(model)
    if explainability_model is None:
        raise ValueError(f"SHAP analysis is not available for final model '{get_model_display_name(model)}'.")

    configure_plot_style()
    x_test = prepared.test_df[prepared.selected_features].values
    if x_test.size == 0:
        raise ValueError("Test set is empty; SHAP analysis cannot be performed.")

    rng = np.random.default_rng(config.random_state)
    sample_size = min(config.shap_sample_limit, x_test.shape[0])
    sample_indices = rng.choice(x_test.shape[0], size=sample_size, replace=False)
    x_sample = x_test[sample_indices]
    sample_df = pd.DataFrame(x_sample, columns=prepared.selected_features)

    explainer = shap.TreeExplainer(explainability_model)
    shap_values = normalize_shap_values(explainer.shap_values(x_sample))

    plt.figure(figsize=(16, 12))
    shap.summary_plot(shap_values, sample_df, show=False, plot_size=(16, 12))
    title_suffix = "" if explainability_mode == "final_model" else " (XGB Base Proxy)"
    plt.title(f"SHAP Summary Plot{title_suffix}", fontsize=30, pad=20)
    ax = plt.gca()
    ax.tick_params(axis="x", labelsize=26)
    ax.tick_params(axis="y", labelsize=26)
    plt.xlabel("SHAP value (impact on model output)", fontsize=30)
    plt.tight_layout()
    plt.savefig(paths.output_dir / "SHAP_Summary_Plot_XGB.png", dpi=600, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, sample_df, plot_type="bar", show=False, plot_size=(12, 8))
    plt.title(f"SHAP Feature Importance ({get_model_display_name(model)}){title_suffix}", fontsize=30, pad=20)
    ax = plt.gca()
    ax.tick_params(axis="x", labelsize=26)
    ax.tick_params(axis="y", labelsize=26)
    plt.tight_layout()
    plt.savefig(paths.output_dir / "SHAP_Feature_Importance_XGB.png", dpi=600, bbox_inches="tight")
    plt.close()

    shap_mean = shap_values.mean(axis=0)
    feature_importance = pd.DataFrame(
        {
            "Feature": prepared.selected_features,
            "SHAP_Importance": np.abs(shap_values).mean(axis=0),
            "SHAP_Mean": shap_mean,
        }
    ).sort_values("SHAP_Importance", ascending=False)
    save_dataframe(feature_importance, paths.output_dir / "SHAP_Feature_Importance_Ranking_XGB.csv")
    save_dataframe(pd.DataFrame(shap_values, columns=prepared.selected_features), paths.output_dir / "SHAP_Values_Matrix_XGB.csv")
    return feature_importance.reset_index(drop=True)


def extract_substructure_smarts(bit_name: str, smiles_library: list[str], config: PipelineConfig) -> str | None:
    """Map a Morgan or MACCS feature name back to an example SMARTS substructure when possible."""
    if str(bit_name).startswith("MACCS_"):
        try:
            return get_maccs_key_smarts(int(str(bit_name).split("_")[-1]))
        except ValueError:
            return None

    if not str(bit_name).startswith("Morgan_"):
        return None

    try:
        bit_index = int(str(bit_name).split("_")[-1])
    except ValueError:
        return None

    if bit_index >= config.morgan_bits:
        return None

    for smiles in smiles_library:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue

        generator = get_morgan_generator(config.morgan_radius, config.morgan_bits)
        additional_output = rdFingerprintGenerator.AdditionalOutput()
        additional_output.AllocateBitInfoMap()
        generator.GetFingerprint(mol, additionalOutput=additional_output)
        bit_info = additional_output.GetBitInfoMap()
        if bit_index not in bit_info:
            continue

        atom_index, radius = bit_info[bit_index][0]
        if radius == 0:
            return mol.GetAtomWithIdx(atom_index).GetSymbol()

        bond_environment = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_index)
        if not bond_environment:
            continue

        submol = Chem.PathToSubmol(mol, bond_environment)
        if submol is not None and submol.GetNumAtoms() > 0:
            return Chem.MolToSmarts(submol)
    return None


def build_top_feature_substructure_table(
    feature_importance: pd.DataFrame,
    prepared: PreparedData,
    config: PipelineConfig,
    top_n: int = 20,
) -> pd.DataFrame:
    """Combine SHAP importance, effect direction and substructure mappings for top features."""
    top_features = feature_importance.head(top_n).copy().reset_index(drop=True)
    smiles_library = prepared.raw_df[config.smiles_col].dropna().astype(str).tolist()
    top_features["Feature_Source"] = top_features["Feature"].apply(
        lambda name: "Morgan" if str(name).startswith("Morgan_") else (
            "MACCS" if str(name).startswith("MACCS_") else (
                "RDKit" if str(name).startswith("RDKit_") else (
                    "Fragment" if str(name).startswith("Frag_") else "Other"
                )
            )
        )
    )
    top_features["Feature_ID"] = top_features.apply(
        lambda row: str(row["Feature"]).split("_", 1)[-1] if row["Feature_Source"] != "Morgan" else str(row["Feature"]).split("_")[-1],
        axis=1,
    )
    top_features["Effect"] = top_features["SHAP_Mean"].apply(lambda value: "Increase" if value >= 0 else "Decrease")
    top_features["Substructure_SMARTS"] = top_features["Feature"].apply(
        lambda feature_name: extract_substructure_smarts(feature_name, smiles_library, config)
    )
    top_features["Display_Label"] = top_features.apply(
        lambda row: f"{row['Feature_Source']}: {row['Feature_ID']}",
        axis=1,
    )
    return top_features[
        [
            "Feature",
            "Feature_Source",
            "Feature_ID",
            "SHAP_Importance",
            "SHAP_Mean",
            "Effect",
            "Substructure_SMARTS",
            "Display_Label",
        ]
    ]


def save_top_feature_substructure_outputs(
    feature_importance: pd.DataFrame,
    prepared: PreparedData,
    paths: PathConfig,
    config: PipelineConfig,
) -> None:
    """Write top-feature substructure tables and structure plots for interpretability."""
    top_features = build_top_feature_substructure_table(feature_importance, prepared, config, top_n=20)
    save_dataframe(top_features, paths.output_dir / "Top20_Feature_Substructure_Mapping.csv")
    plot_top_feature_substructures(top_features, paths.output_dir / "Top20_Feature_Structure_RealData_Fixed.png")


def save_test_error_analysis(
    test_metadata_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Export sample-level held-out test errors with absolute and relative error columns."""
    residuals = y_pred - y_true
    error_df = test_metadata_df.copy()
    error_df["Experimental_em_nm"] = y_true
    error_df["Predicted_em_nm"] = y_pred
    error_df["Residual_Pred_minus_Exp_nm"] = residuals
    error_df["Absolute_Error_nm"] = np.abs(residuals)
    error_df["Squared_Error_nm2"] = residuals ** 2
    error_df = error_df.sort_values("Absolute_Error_nm", ascending=False).reset_index(drop=True)
    error_df["Top5_Absolute_Error_Flag"] = False
    if not error_df.empty:
        top_n = min(5, len(error_df))
        error_df.loc[: top_n - 1, "Top5_Absolute_Error_Flag"] = True
    error_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_run_metadata(
    model: Any,
    prepared: PreparedData,
    config: PipelineConfig,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    """
    Write a JSON record of model type, feature counts, split sizes, metrics and explainability settings.
    """
    explainability_model, explainability_mode = resolve_explainability_model(model)
    feature_summary_df = build_feature_selection_summary(
        prepared.all_df,
        prepared.train_df,
        prepared.test_df,
        prepared.train_f_df,
        prepared.train_fx_df,
        prepared.train_fxx_df,
        final_feature_count=len(prepared.selected_features),
    )
    feature_summary = {
        str(row["Metric"]): int(row["Value"])
        for _, row in feature_summary_df.iterrows()
    }

    payload = {
        "dataset": {
            "total_samples": int(len(prepared.all_df)),
            "train_samples": int(len(prepared.train_df)),
            "test_samples": int(len(prepared.test_df)),
            "target_column": config.target_col,
            "absorbance_column": config.absorb_col,
            "smiles_column": config.smiles_col,
        },
        "fingerprint": {
            "type": "Morgan",
            "radius": config.morgan_radius,
            "n_bits": config.morgan_bits,
        },
        "feature_blocks": {
            "use_morgan_features": config.use_morgan_features,
            "use_maccs_keys": config.use_maccs_keys,
            "use_rdkit_descriptors": config.use_rdkit_descriptors,
            "use_fragment_features": config.use_fragment_features,
        },
        "feature_selection": feature_summary,
        "model": {
            "name": get_model_display_name(model),
            "type": get_model_artifact_type(model),
            "random_state": config.random_state,
            "max_cpu_threads": config.max_cpu_threads,
            "tune_final_xgb": config.tune_final_xgb,
            "final_xgb_tuning_iterations": config.final_xgb_tuning_iterations,
            "final_xgb_tuning_cv_folds": config.final_xgb_tuning_cv_folds,
            "apply_vif_filter": config.apply_vif_filter,
            "vif_threshold": config.vif_threshold,
            "min_features_after_vif": config.min_features_after_vif,
            "rfe_feature_count": len(prepared.rfe_selected_features),
            "final_feature_count": len(prepared.selected_features),
            "vif_removed_feature_count": len(prepared.vif_removed_features),
            "hyperparameters": extract_model_hyperparameters(model),
        },
        "explainability": {
            "method": "TreeSHAP" if explainability_model is not None else "unsupported",
            "source": explainability_mode,
            "source_model": None if explainability_model is None else type(explainability_model).__name__,
        },
        "metrics": metrics,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_model_on_test(
    model: Any,
    prepared: PreparedData,
    paths: PathConfig,
    config: PipelineConfig,
) -> dict[str, float]:
    """
    Evaluate the final model once on the held-out test partition.

    Args:
        model: Fitted final estimator.
        prepared: PreparedData with selected test rows and metadata.
        paths: Output directory for metrics, curves and interpretability files.
        config: Pipeline settings including target column and SHAP limits.

    Returns:
        Regression metrics for the held-out test set.
    """
    x_test = prepared.test_df[prepared.selected_features].values
    y_test = prepared.test_df[config.target_col].values
    y_test_pred = model.predict(x_test)
    metrics = compute_metrics(y_test, y_test_pred)
    model_label = "Mainline Ensemble" if get_model_artifact_type(model) == "mean_ensemble" else get_model_display_name(model)

    metrics_df = pd.DataFrame([metrics])
    save_dataframe(metrics_df, paths.output_dir / "XGB_Test_Set_Metrics.csv")
    save_dataframe(metrics_df, paths.output_dir / "Mainline_Test_Set_Metrics.csv")
    plot_test_regression_curve(
        y_test,
        y_test_pred,
        metrics,
        paths.output_dir / "XGB_Test_Set_Regression_Curve.png",
        title=f"{model_label} Test Set",
    )
    plot_test_regression_curve(
        y_test,
        y_test_pred,
        metrics,
        paths.output_dir / "Mainline_Test_Set_Regression_Curve.png",
        title=f"{model_label} Test Set",
    )
    save_test_error_analysis(
        prepared.test_metadata_df,
        y_test,
        y_test_pred,
        paths.output_dir / "XGB_Test_Set_Error_Analysis.csv",
    )
    save_test_error_analysis(
        prepared.test_metadata_df,
        y_test,
        y_test_pred,
        paths.output_dir / "Mainline_Test_Set_Error_Analysis.csv",
    )
    save_run_metadata(
        model,
        prepared,
        config,
        metrics,
        paths.output_dir / "Pipeline_Run_Metadata.json",
    )

    shap_ranking = run_shap_analysis(model, prepared, paths, config)
    save_top_feature_substructure_outputs(shap_ranking, prepared, paths, config)
    return metrics
