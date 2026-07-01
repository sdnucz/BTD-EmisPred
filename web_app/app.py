from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import traceback
from dataclasses import fields
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import yaml


APP_DIR = Path(__file__).resolve().parent


def find_project_root(app_dir: Path) -> Path:
    for candidate in [app_dir, *app_dir.parents]:
        if (candidate / "emission_project").exists() and (candidate / "data").exists():
            return candidate
    if len(app_dir.parents) >= 3:
        return app_dir.parents[2]
    return app_dir.parent


PROJECT_ROOT = find_project_root(APP_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import PathConfig, PipelineConfig  # noqa: E402
from emission_project.infer import (  # noqa: E402
    build_prediction_features,
    load_prediction_artifacts,
    normalize_smiles_input_columns,
)
from emission_project.utils import (  # noqa: E402
    normalize_solvent_label,
    robust_read_csv,
    smiles_to_morgan,
    smiles_to_mol,
    solvent_to_feature_name,
)


_CONFIG_PATH_VALUE = Path(os.environ.get("MAINLINE_CONFIG_PATH", "config.predict.mainline.yaml"))
CONFIG_PATH = _CONFIG_PATH_VALUE if _CONFIG_PATH_VALUE.is_absolute() else PROJECT_ROOT / _CONFIG_PATH_VALUE
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "mainline"
DEFAULT_MODEL_DISPLAY = "Optuna-XGBoost + OOF-NN correction"
ASSET_DIR = APP_DIR / "assets"
MAX_SMILES_LENGTH = int(os.environ.get("MAX_SMILES_LENGTH", "2048"))
MAX_SOLVENT_LENGTH = int(os.environ.get("MAX_SOLVENT_LENGTH", "120"))
DEBUG_TRACEBACK = os.environ.get("DEBUG_TRACEBACK", "").strip().lower() in {"1", "true", "yes"}


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config_data, dict):
        raise ValueError(f"{config_path.name} must contain a mapping at the top level.")
    return dict(config_data)


def ensure_mapping(section_value: Any, section_name: str) -> dict[str, Any]:
    if section_value is None:
        return {}
    if not isinstance(section_value, dict):
        raise ValueError(f"Section '{section_name}' in the config file must be a mapping.")
    return dict(section_value)


def select_dataclass_fields(
    section_value: dict[str, Any],
    dataclass_type: type[Any],
    section_name: str,
) -> dict[str, Any]:
    valid_fields = {field.name for field in fields(dataclass_type) if field.init}
    unknown_fields = sorted(set(section_value).difference(valid_fields))
    if unknown_fields:
        unknown_text = ", ".join(unknown_fields)
        raise ValueError(f"Unknown keys in config section '{section_name}': {unknown_text}")
    return {name: section_value[name] for name in valid_fields if name in section_value}


def resolve_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def build_path_config(project_root: Path, section_value: dict[str, Any]) -> PathConfig:
    path_values = select_dataclass_fields(section_value, PathConfig, "paths")
    base_dir = resolve_path(project_root, path_values.pop("base_dir", ".")).resolve()
    output_dir = resolve_path(project_root, path_values.pop("output_dir", "outputs/default_run")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return PathConfig(base_dir=base_dir, output_dir=output_dir, **path_values)


def build_pipeline_config(section_value: dict[str, Any]) -> PipelineConfig:
    pipeline_values = select_dataclass_fields(section_value, PipelineConfig, "pipeline")
    return PipelineConfig(**pipeline_values)


def parse_solvent_submission(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if " - " in text:
        text = text.split(" - ", 1)[0].strip()
    text = re.sub(r"\s+\([^)]*\)$", "", text).strip()
    return text


class NNResidualCorrector:
    def __init__(self, output_dir: Path, config: dict[str, Any]) -> None:
        self.output_dir = output_dir
        self.config = dict(config)
        library_path = output_dir / str(self.config.get("library_file", "NN_Residual_Correction_Library.csv"))
        if not library_path.exists():
            raise FileNotFoundError(f"NN residual library not found: {library_path}")

        library = robust_read_csv(library_path)
        smiles_col = str(self.config.get("smiles_column", "SMILES"))
        solvent_col = str(self.config.get("solvent_column", "Solvent"))
        residual_col = str(self.config.get("residual_column", "oof_residual_true_minus_pred"))
        missing = [column for column in (smiles_col, solvent_col, residual_col) if column not in library.columns]
        if missing:
            raise ValueError(f"NN residual library is missing column(s): {', '.join(missing)}")

        library = library.dropna(subset=[smiles_col, residual_col]).reset_index(drop=True)
        if library.empty:
            raise ValueError("NN residual library contains no usable rows.")

        self.radius = int(self.config.get("morgan_radius", 2))
        self.n_bits = int(self.config.get("morgan_bits", 2048))
        self.k = int(self.config.get("k", 10))
        self.shrink = float(self.config.get("shrink", 1.0))
        self.same_solvent = bool(self.config.get("same_solvent", False))
        self.min_max_similarity = float(self.config.get("min_max_similarity", 0.0))
        self.correction_cap_nm = float(self.config.get("correction_cap_nm", -1.0))
        self.enabled = bool(self.config.get("enabled", True))
        self.library_size = int(len(library))
        self.train_bits = np.vstack(
            [
                smiles_to_morgan(smiles, self.radius, self.n_bits).astype(np.float32)
                for smiles in library[smiles_col].astype(str)
            ]
        )
        self.train_bit_counts = self.train_bits.sum(axis=1)
        self.train_solvent = (
            library[solvent_col].map(normalize_solvent_label).fillna("").astype(str).to_numpy()
        )
        self.train_residual = pd.to_numeric(library[residual_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "NNResidualCorrector | None":
        config_path = output_dir / "NN_Residual_Correction_Config.json"
        if not config_path.exists():
            return None
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not bool(config.get("enabled", True)):
            return None
        return cls(output_dir, config)

    def correction_for(self, smiles: str, solvent: str) -> dict[str, Any]:
        if not self.enabled:
            return self._empty_result("disabled")

        query_bits = smiles_to_morgan(smiles, self.radius, self.n_bits).astype(np.float32)
        query_count = float(query_bits.sum())
        if query_count <= 0:
            return self._empty_result("invalid_or_empty_fingerprint")

        candidate_mask = np.ones(self.library_size, dtype=bool)
        if self.same_solvent:
            normalized_solvent = normalize_solvent_label(solvent)
            candidate_mask &= self.train_solvent == normalized_solvent
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            return self._empty_result("no_candidate_neighbors")

        candidate_bits = self.train_bits[candidate_indices]
        intersection = candidate_bits @ query_bits
        union = self.train_bit_counts[candidate_indices] + query_count - intersection
        similarities = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=float),
            where=union > 0,
        )
        max_similarity = float(np.max(similarities)) if similarities.size else 0.0
        if max_similarity < self.min_max_similarity:
            result = self._empty_result("below_similarity_gate")
            result["nn_max_similarity"] = max_similarity
            return result

        neighbor_count = min(max(self.k, 1), int(candidate_indices.size))
        order = np.argsort(-similarities)[:neighbor_count]
        neighbor_similarities = similarities[order]
        neighbor_indices = candidate_indices[order]
        weights = neighbor_similarities + 1e-6
        residual_estimate = float(np.average(self.train_residual[neighbor_indices], weights=weights))
        raw_correction = self.shrink * residual_estimate
        correction = raw_correction
        if self.correction_cap_nm >= 0:
            correction = float(np.clip(correction, -self.correction_cap_nm, self.correction_cap_nm))

        return {
            "nn_correction_enabled": True,
            "nn_gate_active": True,
            "nn_gate_reason": "active",
            "nn_residual_estimate_nm": residual_estimate,
            "nn_residual_correction_nm": correction,
            "nn_raw_residual_correction_nm": raw_correction,
            "nn_max_similarity": max_similarity,
            "nn_mean_neighbor_similarity": float(np.mean(neighbor_similarities)) if neighbor_similarities.size else 0.0,
            "nn_neighbor_count": int(neighbor_count),
            "nn_k": int(self.k),
            "nn_shrink": float(self.shrink),
            "nn_similarity_threshold": float(self.min_max_similarity),
            "nn_correction_cap_nm": float(self.correction_cap_nm),
        }

    def _empty_result(self, reason: str) -> dict[str, Any]:
        return {
            "nn_correction_enabled": self.enabled,
            "nn_gate_active": False,
            "nn_gate_reason": reason,
            "nn_residual_estimate_nm": 0.0,
            "nn_residual_correction_nm": 0.0,
            "nn_raw_residual_correction_nm": 0.0,
            "nn_max_similarity": 0.0,
            "nn_mean_neighbor_similarity": 0.0,
            "nn_neighbor_count": 0,
            "nn_k": int(self.k),
            "nn_shrink": float(self.shrink),
            "nn_similarity_threshold": float(self.min_max_similarity),
            "nn_correction_cap_nm": float(self.correction_cap_nm),
        }


class MainlinePredictor:
    def __init__(self) -> None:
        config_data = load_yaml_config(CONFIG_PATH)
        paths_section = ensure_mapping(config_data.get("paths"), "paths")
        pipeline_section = ensure_mapping(config_data.get("pipeline"), "pipeline")
        self.paths = build_path_config(PROJECT_ROOT, paths_section)
        self.config = build_pipeline_config(pipeline_section)
        self.model, self.selected_features = load_prediction_artifacts(self.paths, self.config)
        self.artifact_metadata = self._load_json(self.paths.artifact_metadata)
        self.experiment_config = self._load_json(self.paths.output_dir / "mainline_nn_experiment_config.json")
        self.nn_corrector = NNResidualCorrector.from_output_dir(self.paths.output_dir)
        self.model_display = self._load_model_display()
        self.metrics = self._load_metrics()
        self.solvent_options = self._load_solvent_options()

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def _load_model_display(self) -> str:
        model_name = self.artifact_metadata.get("model", {}).get("name")
        if model_name:
            return str(model_name)
        return DEFAULT_MODEL_DISPLAY

    def _load_metrics(self) -> dict[str, float]:
        figure_metrics_path = (
            PROJECT_ROOT
            / "outputs"
            / "current_mainline_scatter_bars_features_20260623"
            / "Current_Mainline_Raw_vs_NN_Scatter_Metrics.csv"
        )
        if figure_metrics_path.exists():
            metrics_df = robust_read_csv(figure_metrics_path)
            if not metrics_df.empty and {"dataset", "prediction"}.issubset(metrics_df.columns):
                figure_rows = metrics_df.loc[
                    metrics_df["dataset"].astype(str).eq("heldout_test")
                    & metrics_df["prediction"].astype(str).eq("raw_xgb")
                ]
                if not figure_rows.empty:
                    row = figure_rows.iloc[0]
                    return {
                        "r": round(float(row["r"]), 4),
                        "R2": round(float(row["R2"]), 4),
                        "RMSE": round(float(row["RMSE"]), 1),
                        "MAE": round(float(row["MAE"]), 1),
                    }

        nn_metrics_path = self.paths.output_dir / "mainline_nn_metrics.csv"
        if nn_metrics_path.exists():
            metrics_df = robust_read_csv(nn_metrics_path)
            if not metrics_df.empty and "model_name" in metrics_df.columns:
                corrected_rows = metrics_df.loc[
                    metrics_df["model_name"].astype(str).eq("mainline_xgb_test_nn_corrected")
                ]
                if not corrected_rows.empty:
                    row = corrected_rows.iloc[0]
                    return {
                        key: float(row[key])
                        for key in ("r", "R2", "RMSE", "MAE")
                        if key in metrics_df.columns
                    }

        metric_candidates = [
            self.paths.output_dir / "Mainline_Test_Set_Metrics.csv",
            self.paths.output_dir / "XGB_Test_Set_Metrics.csv",
            MODEL_OUTPUT_DIR / "Mainline_Test_Set_Metrics.csv",
            MODEL_OUTPUT_DIR / "XGB_Test_Set_Metrics.csv",
        ]
        metrics_path = next((path for path in metric_candidates if path.exists()), None)
        if metrics_path is None:
            return {}
        metrics_df = robust_read_csv(metrics_path)
        if metrics_df.empty:
            return {}
        return {
            key: float(metrics_df.iloc[0][key])
            for key in ("r", "R2", "RMSE", "MAE")
            if key in metrics_df.columns
        }

    def _load_solvent_options(self) -> list[str]:
        def display_label(value: Any) -> str:
            label = str(value or "").strip().replace("\\", "/")
            label = re.sub(r"/+", "/", label)
            return label

        def unique_display_options(values: list[str]) -> list[str]:
            options: list[str] = []
            seen: set[str] = set()
            for value in values:
                label = display_label(value)
                key = label.upper()
                if label and key not in seen:
                    options.append(label)
                    seen.add(key)
            return sorted(options, key=lambda item: item.upper())

        raw_path = self.paths.raw_data
        if not raw_path.exists() or not self.config.solvent_col:
            selected = [
                feature.replace("Solvent_", "", 1)
                for feature in self.selected_features
                if feature.startswith("Solvent_")
            ]
            return unique_display_options(selected)

        raw_df = robust_read_csv(raw_path)
        if self.config.solvent_col not in raw_df.columns:
            return []
        normalized = raw_df[self.config.solvent_col].map(normalize_solvent_label)
        values = [value for value in normalized.dropna().astype(str).unique() if value]
        return unique_display_options(values)

    def predict(self, smiles: str, solvent: str) -> dict[str, Any]:
        smiles = str(smiles or "").strip()
        solvent = parse_solvent_submission(solvent)
        normalized_solvent = normalize_solvent_label(solvent)

        if not smiles:
            raise ValueError("SMILES is required.")
        if not normalized_solvent:
            raise ValueError("Solvent is required.")
        if len(smiles) > MAX_SMILES_LENGTH:
            raise ValueError(f"SMILES is too long. Maximum length is {MAX_SMILES_LENGTH} characters.")
        if len(solvent) > MAX_SOLVENT_LENGTH:
            raise ValueError(f"Solvent is too long. Maximum length is {MAX_SOLVENT_LENGTH} characters.")
        if smiles_to_mol(smiles) is None:
            raise ValueError("Invalid SMILES. RDKit cannot parse this structure.")

        input_df = pd.DataFrame(
            [
                {
                    "sample_id": 1,
                    "structure_type": "input",
                    "SMILES": smiles,
                    "Solvent": normalized_solvent,
                }
            ]
        )
        mapping = normalize_smiles_input_columns(input_df)
        feature_df, invalid_smiles = build_prediction_features(
            input_df,
            mapping,
            self.config,
            self.selected_features,
        )
        if invalid_smiles:
            raise ValueError("Invalid SMILES. RDKit cannot parse this structure.")

        x_pred = feature_df[self.selected_features].values
        raw_prediction = float(self.model.predict(x_pred)[0])
        correction_details = (
            self.nn_corrector.correction_for(smiles, normalized_solvent)
            if self.nn_corrector is not None
            else {
                "nn_correction_enabled": False,
                "nn_gate_active": False,
                "nn_gate_reason": "artifact_not_available",
                "nn_residual_correction_nm": 0.0,
                "nn_max_similarity": 0.0,
                "nn_neighbor_count": 0,
            }
        )
        correction = float(correction_details.get("nn_residual_correction_nm", 0.0))
        prediction = raw_prediction + correction
        solvent_feature = solvent_to_feature_name(normalized_solvent)
        warnings: list[str] = []
        if solvent_feature not in self.selected_features:
            warnings.append(
                "This solvent is not one of the selected solvent features in the final model; "
                "the structure features still contribute, but solvent-specific contribution may be limited."
            )
        if self.nn_corrector is not None and not bool(correction_details.get("nn_gate_active", False)):
            warnings.append(
                "Nearest-neighbor residual correction was not applied because the query is outside the fixed similarity gate."
            )

        return {
            "prediction_emission_nm": prediction,
            "prediction_emission_nm_rounded": round(prediction, 2),
            "raw_xgb_prediction_nm": raw_prediction,
            "raw_xgb_prediction_nm_rounded": round(raw_prediction, 2),
            **correction_details,
            "smiles": smiles,
            "solvent": normalized_solvent,
            "model": self.model_display,
            "warnings": warnings,
        }

    def metadata(self) -> dict[str, Any]:
        selection_info = self.experiment_config.get("selection_info", {})
        feature_blocks = self.artifact_metadata.get("feature_blocks", {})
        post_model_correction = self.artifact_metadata.get("post_model_correction", {})
        return {
            "model": "BTD-EmisPred",
            "model_display": self.model_display,
            "model_type": "xgb_nn_residual" if self.nn_corrector is not None else "xgb",
            "target": "NIR-II molecular emission wavelength (nm)",
            "config_path": str(CONFIG_PATH),
            "model_dir": str(self.paths.output_dir),
            "metrics": self.metrics,
            "current_mainline": {
                "dataset_size": self.experiment_config.get("sample_count"),
                "train_size": self.experiment_config.get("train_size"),
                "test_size": self.experiment_config.get("test_size"),
                "split_strategy": self.experiment_config.get("split_strategy"),
                "candidate_feature_count": selection_info.get("initial_feature_count"),
                "after_variance_filter_count": selection_info.get("after_variance_count"),
                "after_feature_screening_count": selection_info.get("after_correlation_count"),
                "selected_feature_count": len(self.selected_features),
                "feature_blocks": feature_blocks,
                "base_model": self.experiment_config.get("base_model", "Optuna-tuned XGBoost"),
                "model_selection": "Eight candidate regressors were compared by 10-fold CV on the training set; XGBoost is served as the final base model.",
                "post_model_correction": post_model_correction,
            },
            "nn_residual_correction": (
                {
                    "enabled": True,
                    "k": self.nn_corrector.k,
                    "shrink": self.nn_corrector.shrink,
                    "min_max_similarity": self.nn_corrector.min_max_similarity,
                    "correction_cap_nm": self.nn_corrector.correction_cap_nm,
                    "library_size": self.nn_corrector.library_size,
                }
                if self.nn_corrector is not None
                else {"enabled": False}
            ),
            "solvent_options": self.solvent_options,
            "solvent_abbreviations": [
                {"abbr": abbr, "full_name": full_name, "display": f"{abbr} - {full_name}"}
                for abbr, full_name in SOLVENT_ABBREVIATIONS
            ],
        }


PREDICTOR: MainlinePredictor | None = None


BASE_STYLE = r"""
:root {
  --ink: #17222f;
  --muted: #647383;
  --line: #d6e0df;
  --field: #ffffff;
  --page: #eef3f1;
  --surface: #fbfcfb;
  --accent: #0b7078;
  --accent-dark: #075b62;
  --copper: #a75536;
  --green: #20785c;
  --red: #b23b4b;
  --amber: #85590e;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
  color: var(--ink);
  background: var(--page);
}
a { color: inherit; }
.topbar {
  background: #ffffff;
  border-bottom: 1px solid var(--line);
}
.topbar-inner {
  max-width: 1160px;
  margin: 0 auto;
  padding: 16px 22px;
  display: flex;
  justify-content: space-between;
  gap: 22px;
  align-items: center;
}
.brand {
  display: flex;
  align-items: center;
  gap: 13px;
  text-decoration: none;
}
.brand-mark {
  width: 64px;
  height: 46px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  background: var(--accent);
  font-weight: 800;
  font-size: 20px;
  line-height: 1;
  white-space: nowrap;
}
.brand-title {
  font-size: 19px;
  font-weight: 800;
  letter-spacing: 0;
}
.brand-subtitle {
  margin-top: 3px;
  color: var(--muted);
  font-size: 12px;
}
.nav {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.nav a {
  text-decoration: none;
  color: #2e3d49;
  font-size: 14px;
  font-weight: 700;
  padding: 9px 11px;
  border-radius: 7px;
}
.nav a:hover, .nav a.active {
  color: white;
  background: var(--accent);
}
.shell {
  max-width: 1160px;
  margin: 0 auto;
  padding: 28px 22px 44px;
}
.page-head {
  position: relative;
  overflow: hidden;
  background:
    linear-gradient(90deg, rgba(255, 255, 255, 0.98), rgba(236, 248, 245, 0.94)),
    repeating-linear-gradient(135deg, rgba(11, 112, 120, 0.06) 0, rgba(11, 112, 120, 0.06) 1px, transparent 1px, transparent 28px);
  border-bottom: 1px solid var(--line);
}
.page-head::after {
  content: "";
  position: absolute;
  right: 0;
  bottom: 0;
  width: 42%;
  height: 100%;
  background: linear-gradient(135deg, transparent 0 46%, rgba(11, 112, 120, 0.08) 46% 54%, transparent 54%);
  pointer-events: none;
}
.page-head-inner {
  position: relative;
  z-index: 1;
  max-width: 1160px;
  margin: 0 auto;
  padding: 46px 22px 42px;
}
.page-head-inner::before {
  content: "";
  position: absolute;
  left: 22px;
  bottom: 24px;
  width: 120px;
  height: 4px;
  border-radius: 999px;
  background: var(--accent);
}
.eyebrow {
  display: inline-flex;
  align-items: center;
  margin: 0 0 12px;
  color: var(--accent-dark);
  background: #e8f5f2;
  border: 1px solid #b7d8d3;
  border-radius: 999px;
  padding: 7px 12px;
  font-size: 13px;
  font-weight: 800;
  text-transform: uppercase;
}
h1 {
  margin: 0;
  font-size: 46px;
  line-height: 1.15;
  letter-spacing: 0;
  color: var(--ink);
}
.lead {
  max-width: 760px;
  margin: 14px 0 0;
  color: #4f6270;
  font-size: 17px;
  line-height: 1.6;
}
.hero-lead {
  max-width: 100%;
  font-size: 15px;
}
.hero-art {
  overflow: hidden;
}
.hero-art img, .image-frame img {
  display: block;
  width: 100%;
  height: auto;
}
.image-frame {
  overflow: hidden;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 10px 24px rgba(27, 45, 53, 0.07);
}
.home-hero {
  grid-template-columns: minmax(300px, 0.85fr) minmax(480px, 1.15fr);
  align-items: stretch;
}
.home-hero > .surface,
.home-hero > .hero-art {
  height: 100%;
}
.home-hero .hero-art {
  overflow: visible;
}
.home-hero .hero-art img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.image-caption {
  margin: 10px 0 0;
  color: var(--muted);
  font-size: 15px;
  font-weight: 700;
  line-height: 1.45;
}
.visual-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}
.guide-step {
  display: grid;
  grid-template-columns: 40px minmax(0, 1fr);
  gap: 14px;
  padding: 16px 0;
  border-top: 1px solid var(--line);
}
.guide-step:first-of-type { border-top: 0; padding-top: 0; }
.step-num {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  display: grid;
  place-items: center;
  background: var(--accent);
  color: white;
  font-weight: 800;
}
.guide-step h3 {
  margin: 0 0 6px;
  font-size: 17px;
}
.guide-step p {
  margin: 0;
}
.abbr-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
.abbr-table th, .abbr-table td {
  border-bottom: 1px solid var(--line);
  padding: 10px 8px;
  text-align: left;
  vertical-align: top;
}
.abbr-table th {
  color: #263541;
  background: #f3f7f6;
}
.abbr-wrap {
  max-height: 430px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
}
.note-panel {
  border-left: 4px solid var(--copper);
  background: #fffaf5;
}
.actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 24px;
}
.button-link, button {
  border: 0;
  border-radius: 7px;
  padding: 13px 16px;
  font-size: 15px;
  font-weight: 800;
  color: white;
  background: var(--accent);
  cursor: pointer;
  text-decoration: none;
  display: inline-flex;
  justify-content: center;
  align-items: center;
  min-height: 46px;
}
.button-link.secondary {
  color: var(--accent-dark);
  background: #ffffff;
  border: 1px solid var(--line);
}
button { width: 100%; }
button:hover, .button-link:hover { background: var(--accent-dark); color: white; }
button:disabled {
  background: #9bb8b8;
  cursor: not-allowed;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 24px;
}
.metric {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 15px;
}
.metric-key {
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 6px;
}
.metric-value {
  font-size: 22px;
  font-weight: 800;
}
.two-col {
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(330px, 0.95fr);
  gap: 22px;
  align-items: start;
}
.three-col {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}
.surface, .card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.surface {
  padding: 24px;
  box-shadow: 0 14px 32px rgba(27, 45, 53, 0.08);
}
.card {
  padding: 20px;
}
.card h3, .surface h2 {
  margin: 0 0 12px;
  font-size: 18px;
  line-height: 1.3;
}
.card p, .surface p, .step-list li {
  color: var(--muted);
  line-height: 1.55;
}
.card p { margin: 0; }
.section-gap { margin-top: 22px; }
label {
  display: block;
  margin-bottom: 8px;
  font-size: 14px;
  font-weight: 700;
  color: #263541;
}
textarea, input {
  width: 100%;
  border: 1px solid #c8d4d2;
  border-radius: 7px;
  padding: 13px 14px;
  font-size: 15px;
  font-family: Arial, Helvetica, sans-serif;
  color: var(--ink);
  background: var(--field);
  outline: none;
}
textarea:focus, input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(11, 112, 120, 0.14);
}
textarea {
  min-height: 172px;
  resize: vertical;
  line-height: 1.48;
}
.field { margin-bottom: 18px; }
.input-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 150px;
  gap: 14px;
  align-items: start;
}
.input-grid > .field {
  margin-bottom: 0;
}
.predict-action {
  padding-top: 27px;
}
.examples {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 10px;
}
.example {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 7px 11px;
  background: #ffffff;
  color: var(--muted);
  font-size: 13px;
  cursor: pointer;
}
.example:hover {
  border-color: var(--accent);
  color: var(--accent-dark);
}
.mail-link {
  color: #0a66c2;
  font-weight: 700;
  text-decoration: none;
}
.mail-link:hover {
  text-decoration: underline;
}
.prediction-block {
  padding: 28px 0 24px;
  border-top: 4px solid var(--accent);
  border-bottom: 1px solid var(--line);
  text-align: center;
}
.prediction-label {
  color: var(--muted);
  font-size: 14px;
  margin-bottom: 10px;
}
.prediction-value {
  color: var(--green);
  font-size: 58px;
  line-height: 1;
  font-weight: 800;
}
.prediction-unit {
  color: var(--muted);
  font-size: 18px;
  font-weight: 700;
  margin-left: 6px;
}
.detail-list {
  margin: 18px 0;
  border-top: 1px solid var(--line);
}
.detail-row {
  display: grid;
  grid-template-columns: 145px minmax(0, 1fr);
  gap: 14px;
  padding: 13px 0;
  border-bottom: 1px solid var(--line);
  align-items: center;
}
.detail-key {
  color: var(--muted);
  font-size: 13px;
}
.detail-value {
  font-size: 15px;
  font-weight: 700;
  overflow-wrap: anywhere;
}
.notice, .error {
  border-radius: 7px;
  padding: 12px 13px;
  font-size: 14px;
  line-height: 1.45;
}
.notice {
  border: 1px solid #e7d4a7;
  background: #fff9eb;
  color: var(--amber);
}
.error {
  border: 1px solid #eabec5;
  background: #fff2f4;
  color: var(--red);
}
.step-list {
  margin: 0;
  padding-left: 20px;
}
footer {
  max-width: 1160px;
  margin: 0 auto;
  padding: 0 22px 28px;
  color: var(--muted);
  font-size: 13px;
}
@media (max-width: 900px) {
  .topbar-inner, .two-col, .three-col { display: block; }
  .nav { justify-content: flex-start; margin-top: 14px; }
  .surface, .card, .metrics { margin-top: 16px; }
  .metrics { grid-template-columns: 1fr; }
  .visual-grid { grid-template-columns: 1fr; }
  .input-grid { grid-template-columns: 1fr; }
  .predict-action { padding-top: 0; }
  .prediction-value { font-size: 46px; }
}
@media (max-width: 540px) {
  .topbar-inner, .shell, .page-head-inner { padding-left: 14px; padding-right: 14px; }
  h1 { font-size: 26px; }
  .brand-subtitle { display: none; }
  .surface { padding: 18px; }
  .detail-row { grid-template-columns: 1fr; gap: 4px; }
}
"""


COMMON_SCRIPT = r"""
<script>
  const metricR2 = document.getElementById("metricR2");
  const metricRmse = document.getElementById("metricRmse");
  const metricMae = document.getElementById("metricMae");
  const smilesEl = document.getElementById("smiles");
  const solventEl = document.getElementById("solvent");
  const buttonEl = document.getElementById("predictBtn");
  const predictionValue = document.getElementById("predictionValue");
  const modelValue = document.getElementById("modelValue");
  const solventValue = document.getElementById("solventValue");
  const rawPredictionValue = document.getElementById("rawPredictionValue");
  const correctionValue = document.getElementById("correctionValue");
  const similarityValue = document.getElementById("similarityValue");
  const statusValue = document.getElementById("statusValue");
  const messageBox = document.getElementById("messageBox");
  const solventOptions = document.getElementById("solventOptions");

  function setText(el, text) {
    if (el) el.textContent = text;
  }

  function setMessage(text, kind = "notice") {
    if (!messageBox) return;
    messageBox.className = kind;
    messageBox.textContent = text;
  }

  function setStatus(text) {
    setText(statusValue, text);
  }

  function normalizeSolventInput(value) {
    const raw = String(value || "").trim();
    if (!raw) return "";
    if (!raw.includes(" - ")) return raw;
    const beforeDash = raw.includes(" - ") ? raw.split(" - ")[0].trim() : raw;
    const mainAbbr = beforeDash.match(/^[A-Za-z0-9/]+/);
    return mainAbbr ? mainAbbr[0] : beforeDash;
  }

  async function loadMetadata() {
    const response = await fetch("/api/metadata");
    const metadata = await response.json();
    const metrics = metadata.metrics || {};
    setText(modelValue, metadata.model_display || metadata.model || "BTD-EmisPred");
    setText(metricR2, metrics.R2 !== undefined ? Number(metrics.R2).toFixed(4) : "--");
    setText(metricRmse, metrics.RMSE !== undefined ? `${Number(metrics.RMSE).toFixed(2)} nm` : "--");
    setText(metricMae, metrics.MAE !== undefined ? `${Number(metrics.MAE).toFixed(2)} nm` : "--");
    if (solventOptions) {
      const seenSolvents = new Set();
      const options = (metadata.solvent_abbreviations || []).map((item) => item.display || "").filter(Boolean);
      const fallbackOptions = metadata.solvent_options || [];
      (options.length ? options : fallbackOptions).forEach((item) => {
        const label = String(item || "").trim().replaceAll("\\", "/");
        const key = label.toUpperCase();
        if (!label || seenSolvents.has(key)) return;
        seenSolvents.add(key);
        const option = document.createElement("option");
        option.value = label;
        solventOptions.appendChild(option);
      });
    }
  }

  async function predict() {
    const smiles = smilesEl ? smilesEl.value.trim() : "";
    const solventInput = solventEl ? solventEl.value.trim() : "";
    const solvent = normalizeSolventInput(solventInput);
    if (!smiles || !solvent) {
      setMessage("SMILES and solvent are both required.", "error");
      setStatus("Missing input");
      return;
    }
    buttonEl.disabled = true;
    buttonEl.textContent = "Predicting...";
    setStatus("Running");
    setMessage("Running prediction...", "notice");
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles, solvent }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Prediction failed.");
      }
      predictionValue.textContent = payload.prediction_emission_nm_rounded.toFixed(2);
      solventValue.textContent = payload.solvent;
      setText(rawPredictionValue, payload.raw_xgb_prediction_nm !== undefined ? `${Number(payload.raw_xgb_prediction_nm).toFixed(2)} nm` : "--");
      setText(correctionValue, payload.nn_residual_correction_nm !== undefined ? `${Number(payload.nn_residual_correction_nm).toFixed(2)} nm` : "--");
      setText(similarityValue, payload.nn_max_similarity !== undefined ? Number(payload.nn_max_similarity).toFixed(3) : "--");
      setStatus("Complete");
      if (payload.warnings && payload.warnings.length) {
        setMessage(payload.warnings.join(" "), "notice");
      } else {
        setMessage("Prediction completed successfully.", "notice");
      }
    } catch (error) {
      predictionValue.textContent = "--";
      solventValue.textContent = "--";
      setText(rawPredictionValue, "--");
      setText(correctionValue, "--");
      setText(similarityValue, "--");
      setStatus("Failed");
      setMessage(error.message, "error");
    } finally {
      buttonEl.disabled = false;
      buttonEl.textContent = "Predict";
    }
  }

  document.querySelectorAll(".example").forEach((el) => {
    el.addEventListener("click", () => {
      if (smilesEl) smilesEl.value = el.dataset.smiles;
      if (solventEl) solventEl.value = el.dataset.solvent;
    });
  });
  if (buttonEl) buttonEl.addEventListener("click", predict);
  if (solventEl) {
    solventEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter") predict();
    });
  }
  loadMetadata().catch((error) => {
    setText(metricR2, "--");
    setText(metricRmse, "--");
    setText(metricMae, "--");
    setMessage(error.message, "error");
  });
</script>
"""


NAV_ITEMS = [
    ("home", "Home", "/home/index.html"),
    ("prediction", "Prediction", "/home/prediction.html"),
    ("explanation", "Method", "/home/explanation.html"),
]


def render_nav(active: str) -> str:
    items = []
    for key, label, href in NAV_ITEMS:
        class_name = ' class="active"' if key == active else ""
        items.append(f'<a href="{href}"{class_name}>{label}</a>')
    return "\n".join(items)


def metric_strip() -> str:
    return """
    <div class="metrics" aria-label="Model performance">
      <div class="metric">
        <div class="metric-key">Held-out test R2</div>
        <div class="metric-value" id="metricR2">--</div>
      </div>
      <div class="metric">
        <div class="metric-key">Held-out test RMSE</div>
        <div class="metric-value" id="metricRmse">--</div>
      </div>
      <div class="metric">
        <div class="metric-key">Held-out test MAE</div>
        <div class="metric-value" id="metricMae">--</div>
      </div>
    </div>
    """


def render_layout(active: str, title: str, subtitle: str, content: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>{BASE_STYLE}</style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <a class="brand" href="/home/index.html">
        <div class="brand-mark">BTD</div>
        <div>
          <div class="brand-title">BTD-EmisPred</div>
          <div class="brand-subtitle">Current mainline NIR-II emission predictor</div>
        </div>
      </a>
      <nav class="nav" aria-label="Main navigation">
        {render_nav(active)}
      </nav>
    </div>
  </header>
  <section class="page-head">
    <div class="page-head-inner">
      <p class="eyebrow">Current Mainline Model</p>
      <h1>{title}</h1>
      <p class="lead">{subtitle}</p>
    </div>
  </section>
  <main class="shell">
    {content}
  </main>
  <footer>
    BTD-EmisPred: Optuna-XGBoost with gated OOF nearest-neighbor residual correction.
  </footer>
  {COMMON_SCRIPT}
</body>
</html>
"""


def render_home_page() -> str:
    content = f"""
    <section class="two-col home-hero">
      <div class="surface">
        <h2>NIR-II emission wavelength prediction</h2>
        <p class="hero-lead">
          This web app serves the current mainline model for solvent-aware NIR-II molecular emission
          wavelength prediction. The model starts from 2095 candidate features, including Morgan
          fingerprints, RDKit fragment counts, and solvent one-hot encoding, then uses training-set
          feature screening to obtain a fixed 82-feature input for deployment.
        </p>
        <p class="hero-lead">
          The deployed predictor is an Optuna-tuned XGBoost model with a gated OOF nearest-neighbor
          residual correction. The result panel reports the base XGBoost prediction, the correction
          term, maximum neighbor similarity, and final emission wavelength.
        </p>
        <div class="actions">
          <a class="button-link" href="/home/prediction.html">Start Prediction</a>
          <a class="button-link secondary" href="/home/explanation.html">View Method</a>
        </div>
      </div>
      <div class="hero-art image-frame">
        <img src="/assets/btd_importance_home_original_v3.png?v=1" alt="BTD-EmisPred model overview visual" />
      </div>
    </section>
    <section class="section-gap">
      {metric_strip()}
    </section>
    <section class="three-col section-gap">
      <div class="card">
        <h3>Input</h3>
        <p>One molecule SMILES and one normalized solvent label.</p>
      </div>
      <div class="card">
        <h3>Mainline</h3>
        <p>Optuna-XGBoost trained on 82 fixed features selected from the training set.</p>
      </div>
      <div class="card">
        <h3>Output</h3>
        <p>Base prediction, OOF-NN correction, similarity evidence, and final wavelength.</p>
      </div>
    </section>
    """
    return render_layout(
        "home",
        "BTD-EmisPred",
        "Current mainline web app for solvent-aware NIR-II molecular emission wavelength prediction.",
        content,
    )


def render_prediction_page() -> str:
    content = """
    <section class="two-col">
      <section class="surface">
        <h2>Prediction Input</h2>
        <div class="field">
          <label for="smiles">SMILES</label>
          <textarea id="smiles" spellcheck="false" placeholder="Paste one molecule SMILES here"></textarea>
        </div>
        <div class="input-grid">
          <div class="field">
            <label for="solvent">Solvent</label>
            <input id="solvent" list="solventOptions" placeholder="THF - Tetrahydrofuran" />
            <datalist id="solventOptions"></datalist>
            <div class="examples">
              <span class="example" data-smiles="CCCCCCCCOC(C=C1)=CC=C1N(C2=CC=C(OCCCCCCCC)C=C2)C(C=C3)=CC=C3C4=CC=C(S4)C5=C(N=[Se]=N6)C6=C(C7=CC=C(C8=CC=C(N(C9=CC=C(OCCCCCCCC)C=C9)C%10=CC=C(OCCCCCCCC)C=C%10)C=C8)S7)C%11=N[Se]N=C%115" data-solvent="TOL (PHME) - Toluene">Example 1</span>
              <span class="example" data-smiles="CN(C)C(C=C1)=CC=C1/C=C/C2=C(OCCO3)C3=C(S2)C(S4)=CC=C4C5=C6N=S=NC6=C(C7=CC=C(C8=C9C(OCCO9)=C(/C=C/C%10=CC=C(N(C)C)C=C%10)S8)S7)C%11=NSN=C%115" data-solvent="H2O - Water">Example 2</span>
            </div>
          </div>
          <div class="field predict-action">
            <button id="predictBtn">Predict</button>
          </div>
        </div>
      </section>
      <section class="surface">
        <h2>Prediction Result</h2>
        <div class="prediction-block">
          <div class="prediction-label">Predicted emission wavelength</div>
          <div>
            <span class="prediction-value" id="predictionValue">--</span>
            <span class="prediction-unit">nm</span>
          </div>
        </div>
        <div class="detail-list">
          <div class="detail-row">
            <div class="detail-key">Model</div>
            <div class="detail-value" id="modelValue">Optuna-XGBoost + OOF-NN correction</div>
          </div>
          <div class="detail-row">
            <div class="detail-key">Solvent</div>
            <div class="detail-value" id="solventValue">--</div>
          </div>
          <div class="detail-row">
            <div class="detail-key">XGBoost base</div>
            <div class="detail-value" id="rawPredictionValue">--</div>
          </div>
          <div class="detail-row">
            <div class="detail-key">OOF-NN correction</div>
            <div class="detail-value" id="correctionValue">--</div>
          </div>
          <div class="detail-row">
            <div class="detail-key">Max similarity</div>
            <div class="detail-value" id="similarityValue">--</div>
          </div>
          <div class="detail-row">
            <div class="detail-key">Status</div>
            <div class="detail-value" id="statusValue">Ready</div>
          </div>
        </div>
        <div id="messageBox" class="notice">Submit a valid SMILES and solvent to run prediction.</div>
      </section>
    </section>
    """
    return render_layout(
        "prediction",
        "Prediction",
        "Enter a molecule SMILES and solvent label; the site returns the current mainline prediction.",
        content,
    )


SOLVENT_ABBREVIATIONS = [
    ("THF", "Tetrahydrofuran"),
    ("DCM", "Dichloromethane"),
    ("TOL (PHME)", "Toluene"),
    ("H2O", "Water"),
    ("DMSO", "Dimethyl sulfoxide"),
    ("MEOH", "Methanol"),
    ("ACN", "Acetonitrile"),
    ("CFM (TCM)", "Chloroform"),
    ("HEX", "n-Hexane"),
    ("ETOH", "Ethanol"),
    ("VAC", "Vinyl acetate"),
    ("ACOET", "Ethyl acetate"),
    ("CHX", "Cyclohexane"),
    ("DMF", "N,N-Dimethylformamide"),
    ("DIOX", "1,4-Dioxane"),
    ("AC", "Acetone"),
    ("IPROPOH (IPA)", "Isopropanol"),
    ("CBZN", "Chlorobenzene"),
    ("BZN", "Benzene"),
    ("SOLID", "Solid state"),
    ("THF/H2O", "THF-Water mixture"),
    ("PBS", "Phosphate buffered saline"),
    ("BZNIT", "Benzonitrile"),
    ("DCE", "1,2-Dichloroethane"),
    ("BLUME", "Butyl methyl ether"),
    ("DEE", "Diethyl ether"),
    ("MXYLENE", "m-Xylene"),
]


def render_solvent_table() -> str:
    rows = "\n".join(
        f"<tr><td>{abbr}</td><td>{full_name}</td></tr>"
        for abbr, full_name in SOLVENT_ABBREVIATIONS
    )
    return f"""
    <div class="abbr-wrap">
      <table class="abbr-table">
        <thead><tr><th>Abbreviation</th><th>Full Name</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """


def render_explanation_page() -> str:
    content = f"""
    <section class="explanation-guide">
      <div class="surface">
        <h2>Current mainline workflow</h2>
        <div class="guide-step">
          <div class="step-num">1</div>
          <div>
            <h3>Data and feature construction</h3>
            <p>The curated dataset contains 1033 molecule-solvent records. Each input is represented by Morgan fingerprints, selected RDKit fragment counts, and solvent one-hot encoding, giving 2095 initial candidate features.</p>
          </div>
        </div>
        <div class="guide-step">
          <div class="step-num">2</div>
          <div>
            <h3>Training-set feature screening</h3>
            <p>The data are split into a fixed 80/20 training-test split. Feature filtering and recursive feature elimination are performed only within the training set, producing a fixed 82-feature mainline input.</p>
          </div>
        </div>
        <div class="guide-step">
          <div class="step-num">3</div>
          <div>
            <h3>Candidate model comparison</h3>
            <p>Eight regressors are compared by 10-fold cross-validation on the training set: CatBoost, XGBoost, LightGBM, RF, GBR, KNN, KRR, and SVR. XGBoost is retained as the final base model.</p>
          </div>
        </div>
        <div class="guide-step">
          <div class="step-num">4</div>
          <div>
            <h3>Final prediction and residual correction</h3>
            <p>The deployed model uses Optuna-tuned XGBoost plus a gated OOF nearest-neighbor residual correction. The correction uses k=10, shrink=1.1, a maximum-similarity gate of 0.5, and a correction cap of 40 nm.</p>
          </div>
        </div>
      </div>
    </section>
    <section class="visual-grid section-gap">
      <div>
        <div class="image-frame">
          <img src="/assets/chemdraw_draw_structure.png" alt="Draw a molecule in ChemDraw" />
        </div>
        <p class="image-caption">Draw the molecule structure in ChemDraw.</p>
      </div>
      <div>
        <div class="image-frame">
          <img src="/assets/chemdraw_copy_smiles.png" alt="Copy structure as SMILES in ChemDraw" />
        </div>
        <p class="image-caption">Select the molecule and copy it as SMILES.</p>
      </div>
      <div>
        <div class="image-frame">
          <img src="/assets/solvent_selection_clear.png?v=2" alt="Select solvent in BTD-EmisPred" />
        </div>
        <p class="image-caption">Input or choose a solvent label from common solvents.</p>
      </div>
      <div>
        <div class="image-frame">
          <img src="/assets/prediction_result_clear.png?v=2" alt="Prediction result in BTD-EmisPred" />
        </div>
        <p class="image-caption">After prediction, the emission wavelength is shown in nm.</p>
      </div>
    </section>
    <section class="two-col section-gap">
      <div class="surface">
        <h2>Solvent abbreviations</h2>
        <p>Please note that only the following solvents use abbreviations:</p>
        {render_solvent_table()}
      </div>
      <div class="surface note-panel">
        <h2>Why are predicted results different from experimental results?</h2>
        <p>
          The discrepancy between the predicted and experimental results can arise from two main aspects.
          First, from the data perspective, the molecule under investigation may contain novel structural
          motifs that are underrepresented or absent in the training set of BTD-EmisPred. Second, from the
          model perspective, all empirical models have intrinsic prediction errors. Our model learns
          statistical correlations from experimental data, which inevitably contain measurement
          uncertainties. Therefore, predicting a perfect match for a new query molecule is challenging.
          We are actively expanding the training database and exploring strategies to quantify prediction
          confidence to better address these limitations.
        </p>
        <p>
          Please let us know your molecule, solvent, and optical properties by sending an email to
          <a class="mail-link" href="mailto:gaowen@sdnu.edu.cn">gaowen@sdnu.edu.cn</a>.
          We will add your molecules to our database as soon as possible.
        </p>
      </div>
    </section>
    """
    return render_layout(
        "explanation",
        "Method",
        "Current mainline model workflow and practical input guidance.",
        content,
    )


PAGE_RENDERERS = {
    "/": render_home_page,
    "/index.html": render_home_page,
    "/home": render_home_page,
    "/home/": render_home_page,
    "/home/index.html": render_home_page,
    "/prediction": render_prediction_page,
    "/prediction.html": render_prediction_page,
    "/home/prediction.html": render_prediction_page,
    "/home/predict.html": render_prediction_page,
    "/explanation": render_explanation_page,
    "/explanation.html": render_explanation_page,
    "/home/explanation.html": render_explanation_page,
}


class AppHandler(BaseHTTPRequestHandler):
    server_version = "MainlineEmissionApp/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), format % args))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_asset(self, relative_path: str) -> None:
        asset_name = Path(relative_path).name
        asset_path = ASSET_DIR / asset_name
        if not asset_path.exists() or not asset_path.is_file():
            self._send_json({"error": "Asset not found."}, HTTPStatus.NOT_FOUND)
            return
        data = asset_path.read_bytes()
        content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/assets/"):
            self._send_asset(parsed.path.removeprefix("/assets/"))
            return
        if parsed.path == "/healthz":
            self._send_json({"status": "ok", "model_loaded": PREDICTOR is not None})
            return
        if parsed.path == "/robots.txt":
            data = b"User-agent: *\nDisallow: /api/\n"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/metadata":
            self._send_json(require_predictor().metadata())
            return
        renderer = PAGE_RENDERERS.get(parsed.path)
        if renderer is not None:
            self._send_html(renderer())
            return
        self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            result = require_predictor().predict(payload.get("smiles", ""), payload.get("solvent", ""))
            self._send_json(result)
        except Exception as exc:
            payload = {"error": str(exc)}
            if DEBUG_TRACEBACK:
                payload["traceback"] = traceback.format_exc(limit=2)
            self._send_json(payload, HTTPStatus.BAD_REQUEST)


def require_predictor() -> MainlinePredictor:
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = MainlinePredictor()
    return PREDICTOR


def run_server(host: str, port: int) -> None:
    predictor = require_predictor()
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"BTD-EmisPred web app is running at http://{host}:{port}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model directory: {predictor.paths.output_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


def run_self_test(smiles: str, solvent: str) -> int:
    result = require_predictor().predict(smiles, solvent)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BTD-EmisPred emission wavelength prediction web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--smiles", default="c1cc(Nc2ccncc2)c2nsnc2c1")
    parser.add_argument("--solvent", default="THF")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test(args.smiles, args.solvent)

    run_server(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
