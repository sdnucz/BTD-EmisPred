"""
Model comparison and final model training utilities.

The comparison stage evaluates candidate regressors by cross-validation on the
training set. The final stage fits the configured mainline model on the complete
training partition and saves the model, selected feature list and metadata needed
for reproducible prediction.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV

from data.dataset import PathConfig, PipelineConfig, PreparedData
from .model import (
    build_final_xgb,
    build_final_mean_ensemble,
    build_final_xgb_param_distributions,
    build_model_specs,
    get_model_artifact_type,
    get_model_display_name,
    save_final_model,
)
from .utils import compute_metrics, plot_cv_regression_curves, save_dataframe


def run_model_comparison(
    prepared: PreparedData,
    paths: PathConfig,
    config: PipelineConfig,
) -> pd.DataFrame:
    """
    Compare candidate regressors using cross-validation inside the training set.

    Args:
        prepared: Prepared training/test data and selected features.
        paths: Output paths for comparison tables and plots.
        config: CV fold counts, random seed and model settings.

    Returns:
        A dataframe summarizing mean and standard deviation of r, R2, RMSE and MAE.

    Key ML flow:
        The outer KFold creates validation folds only from the training partition.
        Inside each outer fold, GridSearchCV tunes hyperparameters on that fold's
        training subset. The held-out project test set is not used in model selection.
    """
    x = prepared.train_model_df[prepared.selected_features].values
    y = prepared.train_model_df[config.target_col].values
    # Model selection happens inside the training partition. The project-level
    # held-out test set is evaluated only after a final model has been chosen.
    outer_cv = KFold(n_splits=config.outer_folds, shuffle=True, random_state=config.random_state)
    model_specs = build_model_specs(config.random_state, config.max_cpu_threads)

    summary_rows: list[dict[str, float | str]] = []
    fold_predictions: dict[str, dict[str, list[float]]] = {
        name: {"y_true": [], "y_pred": []} for name in model_specs
    }

    for model_name, (estimator, param_grid) in model_specs.items():
        fold_metrics: list[dict[str, float]] = []
        for train_index, valid_index in outer_cv.split(x, y):
            x_train, x_valid = x[train_index], x[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            # Inner CV tunes hyperparameters using only the outer-fold training
            # subset; the outer-fold validation subset estimates model-selection
            # performance.
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=config.inner_folds,
                scoring="neg_mean_squared_error",
                n_jobs=1,
            )
            search.fit(x_train, y_train)
            y_pred = search.best_estimator_.predict(x_valid)

            metrics = compute_metrics(y_valid, y_pred)
            fold_metrics.append(metrics)
            fold_predictions[model_name]["y_true"].extend(y_valid.tolist())
            fold_predictions[model_name]["y_pred"].extend(y_pred.tolist())

        summary_rows.append(
            {
                "Model": model_name,
                "Mean r": round(float(np.mean([item["r"] for item in fold_metrics])), 4),
                "Std r": round(float(np.std([item["r"] for item in fold_metrics])), 4),
                "Mean R²": round(float(np.mean([item["R2"] for item in fold_metrics])), 4),
                "Std R²": round(float(np.std([item["R2"] for item in fold_metrics])), 4),
                "Mean RMSE (nm)": round(float(np.mean([item["RMSE"] for item in fold_metrics])), 2),
                "Std RMSE (nm)": round(float(np.std([item["RMSE"] for item in fold_metrics])), 2),
                "Mean MAE (nm)": round(float(np.mean([item["MAE"] for item in fold_metrics])), 2),
                "Std MAE (nm)": round(float(np.std([item["MAE"] for item in fold_metrics])), 2),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("Mean r", ascending=False).reset_index(drop=True)
    save_dataframe(summary_df, paths.output_dir / "Model_Comparison_10FoldCV.csv")
    plot_cv_regression_curves(summary_df, fold_predictions, paths.output_dir)
    return summary_df


def save_final_tuning_results(search: RandomizedSearchCV, paths: PathConfig) -> None:
    """Export the RandomizedSearchCV candidate table and best XGBoost parameters."""
    cv_results_df = pd.DataFrame(search.cv_results_)
    param_columns = [column for column in cv_results_df.columns if column.startswith("param_")]
    export_columns = [
        *param_columns,
        *[
            column
            for column in ["mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time", "std_fit_time"]
            if column in cv_results_df.columns
        ],
    ]
    export_df = cv_results_df[export_columns].copy()
    if "mean_test_score" in export_df.columns:
        export_df["mean_cv_rmse"] = -export_df.pop("mean_test_score")
    if "std_test_score" in export_df.columns:
        export_df["std_cv_rmse"] = export_df.pop("std_test_score")
    if "rank_test_score" in export_df.columns:
        export_df = export_df.sort_values("rank_test_score").reset_index(drop=True)
    save_dataframe(export_df, paths.output_dir / "XGB_Final_Tuning_CV_Results.csv")

    best_payload = {
        "best_params": search.best_params_,
        "best_cv_rmse": float(-search.best_score_),
        "best_rank": 1,
        "n_candidates": int(len(cv_results_df)),
    }
    (paths.output_dir / "XGB_Final_Tuning_Best_Params.json").write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_final_model(prepared: PreparedData, paths: PathConfig, config: PipelineConfig) -> Any:
    """
    Fit the final configured model on the complete training partition.

    Args:
        prepared: PreparedData containing train_model_df and selected features.
        paths: Output paths for optional tuning files.
        config: Final model type and hyperparameter-search settings.

    Returns:
        A fitted estimator. For the mainline workflow this is an XGBRegressor.
    """
    x_train = prepared.train_model_df[prepared.selected_features].values
    y_train = prepared.train_model_df[config.target_col].values
    final_model_type = str(config.final_model_type).strip().lower()

    if final_model_type in {"mean_ensemble", "ensemble_mean"}:
        ensemble_mode = str(config.final_ensemble_mode).strip().lower()
        if ensemble_mode != "mean":
            raise ValueError("Only final_ensemble_mode='mean' is supported for final_model_type='mean_ensemble'.")
        algorithm_names = config.final_ensemble_algorithms or ["XGB", "CAT", "LGBM", "RF"]
        model = build_final_mean_ensemble(
            algorithm_names,
            config.random_state,
            config.max_cpu_threads,
            config.final_xgb_params,
        )
        model.fit(x_train, y_train)
        return model

    if final_model_type != "xgb":
        raise ValueError("pipeline.final_model_type must be 'xgb' or 'mean_ensemble'.")

    if config.tune_final_xgb:
        search = RandomizedSearchCV(
            estimator=build_final_xgb(config.random_state, config.max_cpu_threads, config.final_xgb_params),
            param_distributions=build_final_xgb_param_distributions(),
            n_iter=config.final_xgb_tuning_iterations,
            scoring="neg_root_mean_squared_error",
            cv=KFold(
                n_splits=config.final_xgb_tuning_cv_folds,
                shuffle=True,
                random_state=config.random_state,
            ),
            n_jobs=1,
            random_state=config.random_state,
            refit=True,
            verbose=0,
        )
        search.fit(x_train, y_train)
        save_final_tuning_results(search, paths)
        return search.best_estimator_

    model = build_final_xgb(config.random_state, config.max_cpu_threads, config.final_xgb_params)
    model.fit(x_train, y_train)
    return model


def save_training_artifacts(
    model: Any,
    selected_features: list[str],
    paths: PathConfig,
    config: PipelineConfig,
) -> None:
    """Persist the fitted model, selected feature order and metadata required for prediction."""
    paths.trained_model.parent.mkdir(parents=True, exist_ok=True)
    paths.selected_features_data.parent.mkdir(parents=True, exist_ok=True)
    paths.artifact_metadata.parent.mkdir(parents=True, exist_ok=True)

    save_final_model(model, paths.trained_model)
    save_dataframe(
        pd.DataFrame({"Selected_Features": selected_features}),
        paths.selected_features_data,
    )

    artifact_metadata = {
        "model": {
            "name": get_model_display_name(model),
            "type": get_model_artifact_type(model),
            "path": str(paths.trained_model),
        },
        "selected_features": {
            "count": len(selected_features),
            "path": str(paths.selected_features_data),
        },
        "fingerprint": {
            "morgan_radius": config.morgan_radius,
            "morgan_bits": config.morgan_bits,
        },
        "feature_blocks": {
            "use_morgan_features": config.use_morgan_features,
            "use_maccs_keys": config.use_maccs_keys,
            "use_rdkit_descriptors": config.use_rdkit_descriptors,
            "use_fragment_features": config.use_fragment_features,
            "use_solvent_features": config.use_solvent_features,
            "solvent_col": config.solvent_col,
        },
        "feature_selection": {
            "strategy": "fixed_feature_list" if config.fixed_feature_list_file else "rfe",
            "fixed_feature_list_file": config.fixed_feature_list_file,
        },
        "final_model": {
            "type": config.final_model_type,
            "ensemble_algorithms": config.final_ensemble_algorithms,
            "ensemble_mode": config.final_ensemble_mode,
        },
        "final_xgb_params": config.final_xgb_params,
    }
    paths.artifact_metadata.write_text(
        json.dumps(artifact_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
