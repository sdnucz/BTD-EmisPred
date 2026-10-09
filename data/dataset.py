"""Fold-local data preparation for molecule-grouped nested validation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler
from emission_project.utils import (canonicalize_smiles,normalize_solvent_label,robust_read_csv,
    smiles_to_mol,smiles_to_feature_vector,get_feature_column_names,build_solvent_feature_frame,
    feature_columns,target_columns)
TARGET='λem (nm)'
@dataclass(frozen=True)
class PipelineConfig:
    target_col: str = TARGET
    smiles_col: str = 'SMILES'
    solvent_col: str = 'Solvent'
    absorb_col: str | None = None
    use_solvent_features: bool = True
    deduplicate_smiles: bool = True

def get_optional_absorb_col(config: PipelineConfig) -> str | None:
    """Return the absorbance column name only when it is configured and non-empty."""
    if config.absorb_col is None:
        return None
    absorb_col = str(config.absorb_col).strip()
    return absorb_col or None

def get_optional_solvent_col(config: PipelineConfig) -> str | None:
    """Return the solvent column name only when solvent one-hot features are enabled."""
    if not config.use_solvent_features:
        return None
    if config.solvent_col is None:
        return None
    solvent_col = str(config.solvent_col).strip()
    return solvent_col or None

def validate_raw_dataset(df: pd.DataFrame, config: PipelineConfig) -> None:
    """
    Validate that the raw dataframe has the columns required by the configured model.

    Args:
        df: Input dataframe read from the training CSV.
        config: Pipeline configuration containing column names and feature flags.

    Returns:
        None. Raises ValueError when required columns are missing.
    """
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
    """
    Clean raw rows before featurization.

    Args:
        df: Raw molecule-solvent table.
        config: Pipeline configuration with SMILES, target and solvent column names.

    Returns:
        A filtered dataframe with valid SMILES, numeric emission labels, optional
        numeric absorbance values and normalized solvent labels.
    """
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
    """Return the first non-null value in a grouped metadata series."""
    non_null = series.dropna()
    if non_null.empty:
        return series.iloc[0] if not series.empty else None
    return non_null.iloc[0]

def deduplicate_raw_dataset(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Collapse duplicate molecule records before model training.

    Duplicates are keyed by canonical SMILES and, when solvent features are enabled,
    by the normalized solvent label. Target values are averaged, while metadata uses
    the first non-null value.
    """
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

def variance_filter(train_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Remove zero-variance features using only the training partition."""
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
    """
    Remove highly similar features using only the training partition.

    Features are considered in decreasing variance order. A candidate feature is kept
    only when its maximum absolute correlation to already-kept features is below the
    configured similarity threshold.
    """
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

def group_table(y, groups):
    return pd.DataFrame({'group': groups, 'y': y}).groupby('group', sort=False).y.mean().reset_index()

def stratification(y, bins, minimum):
    count = min(bins, len(np.unique(y)))
    if count < 2:
        return None
    labels = pd.qcut(pd.Series(y), q=count, duplicates='drop')
    if labels.nunique() < 2 or labels.value_counts().min() < minimum:
        return None
    # Preserve the historical implementation's class ordering as well as its seed.
    return labels.astype(str)

def group_holdout(y, groups, seed, test_size, bins):
    table = group_table(y, groups)
    fit, test = train_test_split(np.arange(len(table)), random_state=seed, test_size=test_size,
                                stratify=stratification(table.y, bins, 2))
    return tuple(np.flatnonzero(np.isin(groups, table.group.iloc[index])) for index in (fit, test))

def group_folds(y, groups, seed, folds, bins):
    table = group_table(y, groups)
    labels = stratification(table.y, bins, folds)
    if labels is None:
        indices = KFold(folds, shuffle=True, random_state=seed).split(table)
    else:
        indices = StratifiedKFold(folds, shuffle=True, random_state=seed).split(table, labels)
    return [(np.flatnonzero(np.isin(groups, table.group.iloc[a])),
             np.flatnonzero(np.isin(groups, table.group.iloc[b]))) for a, b in indices]

def featurize(frame, solvent_categories, *, radius, bits):
    if frame.empty or not {'SMILES', 'Solvent'}.issubset(frame):
        raise ValueError('A nonempty SMILES/Solvent table is required.')
    invalid = frame.SMILES.map(lambda s: smiles_to_mol(s) is None)
    solvents = frame.Solvent.map(normalize_solvent_label)
    if invalid.any() or solvents.eq('').any():
        raise ValueError('Invalid SMILES or missing solvent; prediction was not performed.')
    rows = [smiles_to_feature_vector(s, radius, bits, True, False, False, True)[0] for s in frame.SMILES]
    x = pd.DataFrame(np.vstack(rows), columns=get_feature_column_names(bits, True, False, False, True))
    return pd.concat([x, build_solvent_feature_frame(solvents.reset_index(drop=True), solvent_categories)], axis=1)

def check_partition(raw, fit, valid):
    fit, valid = np.asarray(fit, dtype=int), np.asarray(valid, dtype=int)
    if (not len(fit) or not len(valid) or len(np.unique(fit)) != len(fit)
            or len(np.unique(valid)) != len(valid) or min(fit.min(), valid.min()) < 0
            or max(fit.max(), valid.max()) >= len(raw) or np.intersect1d(fit, valid).size):
        raise ValueError('Invalid or overlapping record partitions.')
    if set(raw.iloc[fit].canonical_smiles) & set(raw.iloc[valid].canonical_smiles):
        raise ValueError('Molecules cross the training/validation boundary.')

def feature_matrix(structural, solvents, categories):
    solvents = pd.Series(solvents).reset_index(drop=True).map(normalize_solvent_label)
    if len(structural) != len(solvents) or solvents.eq('').any():
        raise ValueError('Missing solvent or inconsistent input lengths.')
    x = pd.concat([structural.reset_index(drop=True), build_solvent_feature_frame(solvents, categories)], axis=1)
    if x.columns.duplicated().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Duplicate or nonfinite input features.')
    return x, (~solvents.isin(categories)).to_numpy()

def fit_selector(structural, training, config):
    categories = sorted(training.Solvent.map(normalize_solvent_label).unique().tolist())
    x, _ = feature_matrix(structural, training.Solvent, categories)
    y = training[TARGET].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite training labels.')
    frame = x.copy()
    frame[TARGET] = y
    variance, variance_columns, _ = variance_filter(frame)
    correlated, candidate_columns, _ = correlation_filter(variance, config['correlation_threshold'])
    candidates = correlated[candidate_columns]
    if candidates.shape[1] < config['selected_features']:
        raise ValueError(f'Only {candidates.shape[1]} features remain; cannot select {config["selected_features"]}.')
    scaler = StandardScaler()
    selector = RFE(RandomForestRegressor(n_estimators=config['rfe_estimators'],
                   max_depth=config['rfe_depth'], random_state=config['seed'], n_jobs=config['threads']),
                   n_features_to_select=config['selected_features'], step=config['rfe_step'])
    selector.fit(scaler.fit_transform(candidates), y)
    selected = candidates.columns[selector.support_].tolist()
    state = dict(fit_record_ids=training.record_id.astype(int).tolist(),
                 solvent_categories=categories, selected_features=selected,
                 initial_feature_count=x.shape[1], after_variance=variance_columns,
                 after_correlation=candidate_columns, rfe_ranking=selector.ranking_.tolist(),
                 rfe_scaler_mean=scaler.mean_.tolist(), rfe_scaler_scale=scaler.scale_.tolist(),
                 scaling_applied_to='RF-RFE only; estimators receive raw selected features; KNN/KRR/SVR apply their own training-fold scaler',
                 unknown_solvent_policy='all-zero encoding with explicit flag', vif_filter_applied=False)
    return x[selected], y, state

def prepare_fold(structural, raw, fit, valid, config):
    check_partition(raw, fit, valid)
    x_train, y_train, state = fit_selector(structural.iloc[fit], raw.iloc[fit], config)
    x_valid, unknown = feature_matrix(structural.iloc[valid], raw.iloc[valid].Solvent, state['solvent_categories'])
    return dict(x_train=x_train, y_train=y_train, state=state,
                x_valid=x_valid[state['selected_features']],
                y_valid=raw.iloc[valid][TARGET].to_numpy(dtype=float), unknown_solvent=unknown)

def inner_splits(raw, parent, config):
    parent = np.asarray(parent, dtype=int)
    subset = raw.iloc[parent]
    splits = [(parent[a], parent[b]) for a, b in group_folds(
        subset[TARGET].to_numpy(), subset.canonical_smiles.to_numpy(), config['seed'],
        config['inner_folds'], config['stratify_bins'])]
    for fit, valid in splits:
        check_partition(raw, fit, valid)
    if sorted(np.concatenate([b for _, b in splits]).tolist()) != sorted(parent.tolist()):
        raise ValueError('Inner validation coverage is incomplete or duplicated.')
    return splits
def load_data(path):
    config = PipelineConfig()
    source = robust_read_csv(Path(path))
    validate_raw_dataset(source, config)
    raw = deduplicate_raw_dataset(clean_raw_dataset(source, config), config).reset_index(drop=True)
    labels = raw[TARGET].to_numpy(float)
    if not np.isfinite(labels).all() or (labels <= 0).any():
        raise ValueError('Emission wavelengths must be positive and finite.')
    raw['canonical_smiles'] = raw.SMILES.map(canonicalize_smiles)
    raw['record_id'] = np.arange(len(raw))
    return raw
