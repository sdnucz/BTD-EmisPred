from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from emission_project.utils import (
    build_solvent_feature_frame,
    canonicalize_smiles,
    feature_columns,
    get_feature_column_names,
    normalize_solvent_label,
    plot_feature_similarity_heatmap,
    robust_read_csv,
    save_dataframe,
    smiles_to_feature_vector,
    smiles_to_morgan,
    smiles_to_mol,
    target_columns,
)


@dataclass(frozen=True)
class PathConfig:
    base_dir: Path
    output_dir: Path
    raw_data_file: str = "data/data/data.csv"
    prediction_file: str = "data/prediction/SMILES-L.csv"
    trained_model_file: str = "outputs/default_run/XGB_Final_Model.json"
    selected_features_file: str = "outputs/default_run/Final_Model_Selected_Features.csv"
    artifact_metadata_file: str = "outputs/default_run/Model_Artifacts_Metadata.json"

    def _resolve_input_file(self, file_name: str, candidate_dirs: list[str]) -> Path:
        configured_path = Path(file_name)
        if configured_path.is_absolute():
            candidates = [configured_path]
        else:
            candidates = [self.base_dir / configured_path]
            if len(configured_path.parts) == 1:
                candidates.extend(self.base_dir / candidate_dir / configured_path for candidate_dir in candidate_dirs)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _resolve_configured_path(self, file_name: str) -> Path:
        configured_path = Path(file_name)
        if configured_path.is_absolute():
            return configured_path
        return self.base_dir / configured_path

    @property
    def raw_data(self) -> Path:
        return self._resolve_input_file(self.raw_data_file, ["data", "data/data"])

    @property
    def prediction_data(self) -> Path:
        return self._resolve_input_file(self.prediction_file, ["data", "data/prediction"])

    @property
    def trained_model(self) -> Path:
        return self._resolve_configured_path(self.trained_model_file)

    @property
    def selected_features_data(self) -> Path:
        return self._resolve_configured_path(self.selected_features_file)

    @property
    def artifact_metadata(self) -> Path:
        return self._resolve_configured_path(self.artifact_metadata_file)


@dataclass(frozen=True)
class PipelineConfig:
    target_col: str = "λem (nm)"
    absorb_col: str | None = None
    smiles_col: str = "SMILES"
    morgan_radius: int = 2
    morgan_bits: int = 2048
    use_morgan_features: bool = True
    use_maccs_keys: bool = False
    use_rdkit_descriptors: bool = True
    use_fragment_features: bool = True
    use_solvent_features: bool = False
    solvent_col: str | None = None
    test_size: float = 0.2
    stratify_bins: int = 5
    similarity_threshold: float = 0.5
    n_selected_features: int = 60
    random_state: int = 42
    outer_folds: int = 10
    inner_folds: int = 3
    apply_vif_filter: bool = False
    vif_threshold: float = 10.0
    min_features_after_vif: int = 20
    shap_sample_limit: int = 100
    prediction_shap_limit: int = 50
    sample_plot_feature_limit: int = 15
    max_cpu_threads: int = 8
    deduplicate_smiles: bool = True
    fixed_feature_list_file: str | None = None
    final_model_type: str = "xgb"
    final_ensemble_algorithms: list[str] | None = None
    final_ensemble_mode: str = "mean"
    tune_final_xgb: bool = False
    final_xgb_params: dict[str, Any] | None = None
    final_xgb_tuning_iterations: int = 24
    final_xgb_tuning_cv_folds: int = 5
    auto_cleanup_intermediate_outputs: bool = True


@dataclass
class PreparedData:
    raw_df: pd.DataFrame
    metadata_df: pd.DataFrame
    all_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_metadata_df: pd.DataFrame
    test_metadata_df: pd.DataFrame
    train_f_df: pd.DataFrame
    train_fx_df: pd.DataFrame
    train_fxx_df: pd.DataFrame
    train_model_df: pd.DataFrame
    rfe_selected_features: list[str]
    selected_features: list[str]
    vif_removed_features: list[str]


def get_optional_absorb_col(config: PipelineConfig) -> str | None:
    if config.absorb_col is None:
        return None
    absorb_col = str(config.absorb_col).strip()
    return absorb_col or None


def get_optional_solvent_col(config: PipelineConfig) -> str | None:
    if not config.use_solvent_features:
        return None
    if config.solvent_col is None:
        return None
    solvent_col = str(config.solvent_col).strip()
    return solvent_col or None


def get_model_value_columns(config: PipelineConfig) -> list[str]:
    columns: list[str] = []
    absorb_col = get_optional_absorb_col(config)
    if absorb_col is not None:
        columns.append(absorb_col)
    columns.append(config.target_col)
    return columns


def get_metadata_columns(config: PipelineConfig) -> list[str]:
    columns = [config.smiles_col]
    absorb_col = get_optional_absorb_col(config)
    if absorb_col is not None:
        columns.append(absorb_col)
    solvent_col = get_optional_solvent_col(config)
    if solvent_col is not None:
        columns.append(solvent_col)
    columns.append(config.target_col)
    return columns


def validate_raw_dataset(df: pd.DataFrame, config: PipelineConfig) -> None:
    required_columns = {config.smiles_col, config.target_col}
    absorb_col = get_optional_absorb_col(config)
    if absorb_col is not None:
        required_columns.add(absorb_col)
    if config.use_solvent_features:
        solvent_col = get_optional_solvent_col(config)
        if solvent_col is None:
            raise ValueError("pipeline.solvent_col must be set when pipeline.use_solvent_features is true.")
        required_columns.add(solvent_col)
    missing = required_columns.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Raw dataset is missing required columns: {missing_str}")


def clean_raw_dataset(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    cleaned_df = df.copy()
    cleaned_df[config.smiles_col] = cleaned_df[config.smiles_col].astype("string").str.strip()
    cleaned_df[config.target_col] = pd.to_numeric(cleaned_df[config.target_col], errors="coerce")

    valid_mask = cleaned_df[config.smiles_col].notna() & cleaned_df[config.smiles_col].ne("")
    valid_mask &= cleaned_df[config.target_col].notna()
    valid_mask &= cleaned_df[config.smiles_col].map(lambda smiles: smiles_to_mol(smiles) is not None)

    absorb_col = get_optional_absorb_col(config)
    if absorb_col is not None:
        cleaned_df[absorb_col] = pd.to_numeric(cleaned_df[absorb_col], errors="coerce")
        valid_mask &= cleaned_df[absorb_col].notna()

    solvent_col = get_optional_solvent_col(config)
    if solvent_col is not None:
        cleaned_df[solvent_col] = cleaned_df[solvent_col].map(normalize_solvent_label)
        valid_mask &= cleaned_df[solvent_col].notna() & cleaned_df[solvent_col].astype("string").ne("")

    cleaned_df = cleaned_df.loc[valid_mask].reset_index(drop=True)
    if cleaned_df.empty:
        raise ValueError("Raw dataset has no valid rows after filtering missing or non-numeric target values.")
    return cleaned_df


def first_non_null_value(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return series.iloc[0] if not series.empty else None
    return non_null.iloc[0]


def deduplicate_raw_dataset(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    if not config.deduplicate_smiles:
        return df.reset_index(drop=True)

    dedup_df = df.copy()
    dedup_df["_canonical_smiles"] = dedup_df[config.smiles_col].map(canonicalize_smiles)
    solvent_col = get_optional_solvent_col(config)
    if solvent_col is None:
        dedup_df["_dedup_key"] = dedup_df["_canonical_smiles"].fillna(dedup_df[config.smiles_col])
    else:
        dedup_df["_dedup_key"] = (
            dedup_df["_canonical_smiles"].fillna(dedup_df[config.smiles_col]).astype(str)
            + "||"
            + dedup_df[solvent_col].astype(str)
        )

    if dedup_df["_dedup_key"].is_unique:
        return dedup_df.drop(columns=["_canonical_smiles", "_dedup_key"]).reset_index(drop=True)

    aggregation_map = {
        column: first_non_null_value
        for column in dedup_df.columns
        if column not in {"_canonical_smiles", "_dedup_key", config.target_col}
    }
    aggregation_map[config.target_col] = "mean"

    absorb_col = get_optional_absorb_col(config)
    if absorb_col is not None:
        aggregation_map[absorb_col] = "mean"

    aggregated_df = dedup_df.groupby("_dedup_key", sort=False, as_index=False).agg(aggregation_map)
    return aggregated_df.drop(columns=["_dedup_key"]).reset_index(drop=True)


def build_feature_tables(
    raw_df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows = [
        smiles_to_feature_vector(
            smiles,
            config.morgan_radius,
            config.morgan_bits,
            config.use_morgan_features,
            config.use_maccs_keys,
            config.use_rdkit_descriptors,
            config.use_fragment_features,
        )[0]
        for smiles in raw_df[config.smiles_col]
    ]
    feature_df = pd.DataFrame(
        np.vstack(feature_rows),
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
        solvent_feature_df = build_solvent_feature_frame(raw_df[solvent_col])
        feature_df = pd.concat([feature_df, solvent_feature_df], axis=1)
    model_value_columns = get_model_value_columns(config)
    metadata_columns = get_metadata_columns(config)
    model_df = pd.concat(
        [feature_df, raw_df[model_value_columns].reset_index(drop=True)],
        axis=1,
    )
    helper_df = pd.concat(
        [
            feature_df,
            raw_df[metadata_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    return model_df, helper_df


def stratified_split(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    indices = np.arange(len(df))
    stratify_bins = min(config.stratify_bins, df[config.target_col].nunique())
    if stratify_bins < 2:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=config.test_size,
            shuffle=True,
            random_state=config.random_state,
        )
    else:
        stratify_labels = pd.qcut(df[config.target_col], q=stratify_bins, duplicates="drop")
        train_indices, test_indices = train_test_split(
            indices,
            test_size=config.test_size,
            shuffle=True,
            random_state=config.random_state,
            stratify=stratify_labels,
        )
    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    return train_df, test_df, np.asarray(train_indices), np.asarray(test_indices)


def variance_filter(train_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    features = feature_columns(train_df)
    targets = target_columns(train_df)
    variances = train_df[features].var()
    kept_features = variances[variances > 0].index.tolist()
    dropped_features = [feature for feature in features if feature not in kept_features]
    reduced_df = pd.concat([train_df[kept_features], train_df[targets]], axis=1)
    return reduced_df, kept_features, dropped_features


def correlation_filter(
    train_df: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    features = feature_columns(train_df)
    targets = target_columns(train_df)
    corr_matrix = train_df[features].corr().abs()
    variance_order = train_df[features].var().sort_values(ascending=False).index.tolist()

    selected_features: list[str] = []
    dropped_features: list[str] = []
    for feature in variance_order:
        if not selected_features:
            selected_features.append(feature)
            continue

        max_similarity = float(corr_matrix.loc[feature, selected_features].max())
        if np.isnan(max_similarity) or max_similarity < threshold:
            selected_features.append(feature)
        else:
            dropped_features.append(feature)

    reduced_df = pd.concat([train_df[selected_features], train_df[targets]], axis=1)
    return reduced_df, selected_features, dropped_features


def rfe_select_features(
    train_df: pd.DataFrame,
    target_col: str,
    n_selected_features: int,
    random_state: int,
    max_cpu_threads: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    features = feature_columns(train_df)
    targets = target_columns(train_df)
    n_selected = min(n_selected_features, len(features))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(train_df[features].values)
    y = train_df[target_col].values

    estimator = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=random_state,
        n_jobs=max_cpu_threads,
    )
    selector = RFE(
        estimator=estimator,
        n_features_to_select=n_selected,
        step=2,
        verbose=0,
    )
    selector.fit(x_scaled, y)

    selected_features = [features[index] for index, keep in enumerate(selector.support_) if keep]
    dropped_features = [features[index] for index, keep in enumerate(selector.support_) if not keep]
    reduced_df = pd.concat([train_df[selected_features], train_df[targets]], axis=1)
    return reduced_df, selected_features, dropped_features


def build_sample_metadata(raw_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    metadata_df = raw_df[get_metadata_columns(config)].copy().reset_index(drop=True)
    metadata_df.insert(0, "Sample_ID", np.arange(1, len(metadata_df) + 1))
    return metadata_df


def resolve_fixed_feature_list_path(paths: PathConfig, config: PipelineConfig) -> Path | None:
    if config.fixed_feature_list_file is None:
        return None
    configured_path = Path(str(config.fixed_feature_list_file))
    if configured_path.is_absolute():
        return configured_path
    return paths.base_dir / configured_path


def load_fixed_feature_list(feature_list_path: Path, available_features: list[str]) -> list[str]:
    if not feature_list_path.exists():
        raise FileNotFoundError(f"Fixed feature list file not found: {feature_list_path}")

    feature_df = robust_read_csv(feature_list_path)
    if feature_df.empty:
        raise ValueError(f"Fixed feature list file is empty: {feature_list_path}")

    column_name = "Selected_Features" if "Selected_Features" in feature_df.columns else feature_df.columns[0]
    selected_features = feature_df[column_name].dropna().astype(str).tolist()
    if not selected_features:
        raise ValueError(f"Fixed feature list file contains no usable features: {feature_list_path}")

    selected_features = list(dict.fromkeys(selected_features))
    available_feature_set = set(available_features)
    missing_features = [feature_name for feature_name in selected_features if feature_name not in available_feature_set]
    if missing_features:
        preview = ", ".join(missing_features[:5])
        raise ValueError(
            "Fixed feature list contains features unavailable after the current variance/correlation filters. "
            f"Missing {len(missing_features)} features. Examples: {preview}"
        )
    return selected_features


def build_feature_selection_summary(
    all_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_f_df: pd.DataFrame,
    train_fx_df: pd.DataFrame,
    train_fxx_df: pd.DataFrame,
    final_feature_count: int | None = None,
) -> pd.DataFrame:
    initial_feature_count = len(feature_columns(all_df))
    variance_feature_count = len(feature_columns(train_f_df))
    correlation_feature_count = len(feature_columns(train_fx_df))
    rfe_feature_count = len(feature_columns(train_fxx_df))
    model_feature_count = rfe_feature_count if final_feature_count is None else final_feature_count

    metrics = [
        "Total_Samples",
        "Train_Samples",
        "Test_Samples",
        "Initial_Features",
        "After_Variance_Filter",
        "Removed_By_Variance_Filter",
        "After_Correlation_Filter",
        "Removed_By_Correlation_Filter",
        "After_RFE",
        "Removed_By_RFE",
    ]
    values = [
        len(all_df),
        len(train_df),
        len(test_df),
        initial_feature_count,
        variance_feature_count,
        initial_feature_count - variance_feature_count,
        correlation_feature_count,
        variance_feature_count - correlation_feature_count,
        rfe_feature_count,
        correlation_feature_count - rfe_feature_count,
    ]

    if model_feature_count != rfe_feature_count:
        metrics.extend([
            "After_VIF_Filter",
            "Removed_By_VIF_Filter",
        ])
        values.extend([
            model_feature_count,
            rfe_feature_count - model_feature_count,
        ])

    return pd.DataFrame({"Metric": metrics, "Value": values})


def compute_vif_summary(df: pd.DataFrame) -> pd.DataFrame:
    features = feature_columns(df)
    if not features:
        return pd.DataFrame(columns=["Feature", "VIF", "Tolerance", "Max_Abs_PairCorr", "VIF_Level"])

    x = df[features].astype(float).values
    x_scaled = StandardScaler().fit_transform(x)

    if len(features) == 1:
        return pd.DataFrame(
            {
                "Feature": features,
                "VIF": [1.0],
                "Tolerance": [1.0],
                "Max_Abs_PairCorr": [0.0],
                "VIF_Level": ["Low"],
            }
        )

    corr_matrix = np.abs(np.corrcoef(x_scaled, rowvar=False))
    np.fill_diagonal(corr_matrix, 0.0)
    max_pair_corr = corr_matrix.max(axis=1)

    vif_rows: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(features):
        y_feature = x_scaled[:, feature_index]
        x_other = np.delete(x_scaled, feature_index, axis=1)
        model = LinearRegression()
        model.fit(x_other, y_feature)
        r2_value = float(model.score(x_other, y_feature))

        if r2_value >= 0.999999:
            vif_value = float("inf")
            tolerance = 0.0
        else:
            tolerance = max(1.0 - r2_value, 0.0)
            vif_value = 1.0 / tolerance if tolerance > 0 else float("inf")

        if vif_value >= 10:
            vif_level = "High"
        elif vif_value >= 5:
            vif_level = "Moderate"
        else:
            vif_level = "Low"

        vif_rows.append(
            {
                "Feature": feature_name,
                "VIF": vif_value,
                "Tolerance": tolerance,
                "Max_Abs_PairCorr": float(max_pair_corr[feature_index]),
                "VIF_Level": vif_level,
            }
        )

    return pd.DataFrame(vif_rows).sort_values(
        ["VIF", "Max_Abs_PairCorr"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def vif_filter_features(
    train_df: pd.DataFrame,
    vif_threshold: float,
    min_features_after_vif: int,
) -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame, pd.DataFrame]:
    features = feature_columns(train_df)
    targets = target_columns(train_df)
    current_features = features.copy()
    removed_features: list[str] = []
    refinement_rows: list[dict[str, Any]] = []

    while len(current_features) > max(1, min_features_after_vif):
        current_df = pd.concat([train_df[current_features], train_df[targets]], axis=1)
        vif_summary = compute_vif_summary(current_df)
        if vif_summary.empty:
            break

        top_row = vif_summary.iloc[0]
        top_vif = float(top_row["VIF"])
        if np.isfinite(top_vif) and top_vif <= vif_threshold:
            break

        feature_to_remove = str(top_row["Feature"])
        refinement_rows.append(
            {
                "Step": len(refinement_rows) + 1,
                "Removed_Feature": feature_to_remove,
                "Removed_Feature_VIF": top_vif,
                "Removed_Feature_Max_Abs_PairCorr": float(top_row["Max_Abs_PairCorr"]),
                "Remaining_Features_After_Removal": len(current_features) - 1,
            }
        )
        current_features.remove(feature_to_remove)
        removed_features.append(feature_to_remove)

    reduced_df = pd.concat([train_df[current_features], train_df[targets]], axis=1)
    final_vif_summary = compute_vif_summary(reduced_df)
    refinement_log = pd.DataFrame(refinement_rows)
    return reduced_df, current_features, removed_features, refinement_log, final_vif_summary


def prepare_datasets(paths: PathConfig, config: PipelineConfig) -> PreparedData:
    raw_df = robust_read_csv(paths.raw_data)
    validate_raw_dataset(raw_df, config)
    raw_df = clean_raw_dataset(raw_df, config)
    raw_df = deduplicate_raw_dataset(raw_df, config)
    metadata_df = build_sample_metadata(raw_df, config)

    all_df, helper_df = build_feature_tables(raw_df, config)
    train_df, test_df, train_indices, test_indices = stratified_split(all_df, config)
    train_metadata_df = metadata_df.iloc[train_indices].reset_index(drop=True)
    test_metadata_df = metadata_df.iloc[test_indices].reset_index(drop=True)
    train_f_df, _, _ = variance_filter(train_df)
    train_fx_df, _, _ = correlation_filter(train_f_df, config.similarity_threshold)
    fixed_feature_list_path = resolve_fixed_feature_list_path(paths, config)
    if fixed_feature_list_path is None:
        train_fxx_df, rfe_selected_features, _ = rfe_select_features(
            train_fx_df,
            config.target_col,
            config.n_selected_features,
            config.random_state,
            config.max_cpu_threads,
        )
    else:
        rfe_selected_features = load_fixed_feature_list(fixed_feature_list_path, feature_columns(train_df))
        train_fxx_df = pd.concat([train_df[rfe_selected_features], train_df[target_columns(train_df)]], axis=1)
    initial_vif_summary = compute_vif_summary(train_fxx_df)

    if config.apply_vif_filter:
        train_model_df, selected_features, vif_removed_features, vif_refinement_log, final_vif_summary = vif_filter_features(
            train_fxx_df,
            config.vif_threshold,
            config.min_features_after_vif,
        )
    else:
        train_model_df = train_fxx_df.copy()
        selected_features = rfe_selected_features.copy()
        vif_removed_features = []
        vif_refinement_log = pd.DataFrame(
            columns=[
                "Step",
                "Removed_Feature",
                "Removed_Feature_VIF",
                "Removed_Feature_Max_Abs_PairCorr",
                "Remaining_Features_After_Removal",
            ]
        )
        final_vif_summary = initial_vif_summary.copy()

    split_assignment_df = metadata_df.copy()
    split_assignment_df["Split"] = "Train"
    split_assignment_df.loc[test_indices, "Split"] = "Test"
    feature_selection_summary = build_feature_selection_summary(
        all_df,
        train_df,
        test_df,
        train_f_df,
        train_fx_df,
        train_fxx_df,
        final_feature_count=len(selected_features),
    )

    save_dataframe(all_df, paths.output_dir / "Feature_all.csv")
    save_dataframe(helper_df, paths.output_dir / "Feature_all_with_smiles.csv")
    save_dataframe(train_df, paths.output_dir / "Feature_train.csv")
    save_dataframe(test_df, paths.output_dir / "Feature_test.csv")
    save_dataframe(train_f_df, paths.output_dir / "Feature_train_F.csv")
    save_dataframe(train_fx_df, paths.output_dir / "Feature_train_FX.csv")
    save_dataframe(train_fxx_df, paths.output_dir / "Feature_train_FXX.csv")
    save_dataframe(pd.DataFrame({"Selected_Features": rfe_selected_features}), paths.output_dir / "FXX_selected_features.csv")
    save_dataframe(pd.DataFrame({"Selected_Features": selected_features}), paths.output_dir / "Final_Model_Selected_Features.csv")
    save_dataframe(split_assignment_df, paths.output_dir / "Dataset_Split_Assignment.csv")
    save_dataframe(feature_selection_summary, paths.output_dir / "Feature_Selection_Summary.csv")
    save_dataframe(initial_vif_summary, paths.output_dir / "Feature_VIF_Summary_Before_Filter.csv")
    save_dataframe(final_vif_summary, paths.output_dir / "Feature_VIF_Summary.csv")

    if config.apply_vif_filter:
        save_dataframe(train_model_df, paths.output_dir / "Feature_train_FXX_VIF.csv")
        save_dataframe(pd.DataFrame({"Removed_Feature": vif_removed_features}), paths.output_dir / "VIF_Removed_Features.csv")
        save_dataframe(vif_refinement_log, paths.output_dir / "Feature_VIF_Refinement_Log.csv")

    plot_feature_similarity_heatmap(train_fxx_df, paths.output_dir / "Feature_Similarity_Heatmap_English_LargeFont.png")

    return PreparedData(
        raw_df=raw_df,
        metadata_df=metadata_df,
        all_df=all_df,
        train_df=train_df,
        test_df=test_df,
        train_metadata_df=train_metadata_df,
        test_metadata_df=test_metadata_df,
        train_f_df=train_f_df,
        train_fx_df=train_fx_df,
        train_fxx_df=train_fxx_df,
        train_model_df=train_model_df,
        rfe_selected_features=rfe_selected_features,
        selected_features=selected_features,
        vif_removed_features=vif_removed_features,
    )
