from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


TRAIN_INTERMEDIATE_FILENAMES = {
    "Morgan_all.csv",
    "Morgan_all_with_smiles.csv",
    "Morgan_train.csv",
    "Morgan_test.csv",
    "Morgan_train_F.csv",
    "Morgan_train_FX.csv",
    "Morgan_train_FXX.csv",
    "Morgan_train_FXX_VIF.csv",
    "Feature_all.csv",
    "Feature_all_with_smiles.csv",
    "Feature_train.csv",
    "Feature_test.csv",
    "Feature_train_F.csv",
    "Feature_train_FX.csv",
    "Feature_train_FXX.csv",
    "Feature_train_FXX_VIF.csv",
    "FXX_selected_features.csv",
    "SHAP_Values_Matrix_XGB.csv",
}


def cleanup_intermediate_outputs(output_dir: Path, stage: str) -> list[Path]:
    if stage not in {"final", "all", "predict"}:
        return []

    filenames_to_delete = set()
    if stage in {"final", "all"}:
        filenames_to_delete.update(TRAIN_INTERMEDIATE_FILENAMES)

    deleted_paths: list[Path] = []
    for file_name in sorted(filenames_to_delete):
        candidate = output_dir / file_name
        if not candidate.exists():
            continue
        candidate.unlink()
        deleted_paths.append(candidate)
    return deleted_paths


def prune_batch_run_outputs(
    batch_dir: Path,
    keep_top_n: int = 5,
    cleanup_intermediate_files: bool = True,
) -> dict[str, object]:
    summary_path = batch_dir / "final_run_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Batch summary not found: {summary_path}")

    summary_df = pd.read_csv(summary_path, encoding="utf-8-sig")
    if summary_df.empty:
        raise ValueError(f"Batch summary is empty: {summary_path}")

    keep_count = max(1, min(int(keep_top_n), len(summary_df)))
    ranked_df = summary_df.sort_values(["R2", "r", "seed"], ascending=[False, False, True]).reset_index(drop=True)
    ranked_df.to_csv(batch_dir / "final_run_summary_full.csv", index=False, encoding="utf-8-sig")
    retained_df = ranked_df.head(keep_count).copy().reset_index(drop=True)

    retained_run_dirs = {str(run_dir) for run_dir in retained_df["run_dir"].tolist()}
    removed_run_dirs: list[str] = []
    for candidate in sorted(batch_dir.iterdir()):
        if not candidate.is_dir() or not candidate.name.startswith("seed_"):
            continue
        if candidate.name in retained_run_dirs:
            if cleanup_intermediate_files:
                cleanup_intermediate_outputs(candidate, "final")
            continue
        shutil.rmtree(candidate)
        removed_run_dirs.append(candidate.name)

    best_run_dir = batch_dir / "best_run"
    if best_run_dir.exists() and cleanup_intermediate_files:
        cleanup_intermediate_outputs(best_run_dir, "final")

    retained_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    best_row = retained_df.iloc[0]
    mean_final_feature_count = None
    if "final_feature_count" in retained_df.columns:
        mean_final_feature_count = float(retained_df["final_feature_count"].mean())

    mean_vif_removed_feature_count = None
    if "vif_removed_feature_count" in retained_df.columns:
        mean_vif_removed_feature_count = float(retained_df["vif_removed_feature_count"].mean())

    stats_payload = {
        "run_count": int(len(retained_df)),
        "best_seed": int(best_row["seed"]),
        "best_R2": float(best_row["R2"]),
        "best_RMSE": float(best_row["RMSE"]),
        "best_MAE": float(best_row["MAE"]),
        "mean_R2": float(retained_df["R2"].mean()),
        "median_R2": float(retained_df["R2"].median()),
        "mean_RMSE": float(retained_df["RMSE"].mean()),
        "median_RMSE": float(retained_df["RMSE"].median()),
        "mean_MAE": float(retained_df["MAE"].mean()),
        "median_MAE": float(retained_df["MAE"].median()),
        "mean_final_feature_count": mean_final_feature_count,
        "mean_vif_removed_feature_count": mean_vif_removed_feature_count,
        "original_run_count": int(len(summary_df)),
        "full_summary_file": "final_run_summary_full.csv",
        "full_mean_R2": float(ranked_df["R2"].mean()),
        "full_median_R2": float(ranked_df["R2"].median()),
        "full_mean_RMSE": float(ranked_df["RMSE"].mean()),
        "full_median_RMSE": float(ranked_df["RMSE"].median()),
        "full_mean_MAE": float(ranked_df["MAE"].mean()),
        "full_median_MAE": float(ranked_df["MAE"].median()),
        "retained_run_count": int(len(retained_df)),
        "retained_run_dirs": retained_df["run_dir"].tolist(),
        "removed_run_count": int(len(removed_run_dirs)),
        "removed_run_dirs": removed_run_dirs,
    }

    stats_path = batch_dir / "batch_statistics.json"
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "original_run_count": int(len(summary_df)),
        "retained_run_count": int(len(retained_df)),
        "retained_run_dirs": retained_df["run_dir"].tolist(),
        "removed_run_count": int(len(removed_run_dirs)),
        "removed_run_dirs": removed_run_dirs,
    }
