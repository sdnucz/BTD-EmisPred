"""
Command-line entry point for the BTD-EmisPred workflow.

The module reads the YAML configuration, limits CPU resources, dispatches the
requested stage, and connects data preparation, model comparison, final model
training, held-out testing and batch prediction.
"""
from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyYAML is required to load config.YAML. Install it with 'python -m pip install PyYAML'."
    ) from exc

from data.dataset import PathConfig, PipelineConfig, prepare_datasets

from .cleanup import cleanup_intermediate_outputs
from .infer import load_prediction_artifacts, run_prediction_shap_analysis, run_prediction_workflow
from .test import evaluate_model_on_test
from .train import run_model_comparison, save_training_artifacts, train_final_model


DEFAULT_CONFIG_PATH = "config.YAML"
CONFIG_ENV_VAR = "EMISSION_CONFIG"
VALID_STAGES = {"prepare", "compare", "final", "predict", "all"}


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Read a YAML configuration file and return its top-level mapping."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config_data, dict):
        raise ValueError(f"{config_path.name} must contain a mapping at the top level.")
    return dict(config_data)


def ensure_mapping(section_value: Any, section_name: str) -> dict[str, Any]:
    """Normalize an optional YAML section to a dictionary and reject invalid section types."""
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
    """
    Filter a YAML mapping to fields accepted by a dataclass constructor and reject unknown keys.
    """
    valid_fields = {field.name for field in fields(dataclass_type) if field.init}
    unknown_fields = sorted(set(section_value).difference(valid_fields))
    if unknown_fields:
        unknown_text = ", ".join(unknown_fields)
        raise ValueError(f"Unknown keys in config section '{section_name}': {unknown_text}")
    return {name: section_value[name] for name in valid_fields if name in section_value}


def resolve_path(project_root: Path, path_value: str | Path) -> Path:
    """Resolve an absolute path or a project-root-relative path."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def build_path_config(project_root: Path, section_value: dict[str, Any]) -> PathConfig:
    """Construct PathConfig from the YAML paths section and create the output directory."""
    path_values = select_dataclass_fields(section_value, PathConfig, "paths")
    base_dir = resolve_path(project_root, path_values.pop("base_dir", ".")).resolve()
    output_dir = resolve_path(project_root, path_values.pop("output_dir", "outputs/default_run")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return PathConfig(base_dir=base_dir, output_dir=output_dir, **path_values)


def build_pipeline_config(section_value: dict[str, Any]) -> PipelineConfig:
    """Construct PipelineConfig from the YAML pipeline section."""
    pipeline_values = select_dataclass_fields(section_value, PipelineConfig, "pipeline")
    return PipelineConfig(**pipeline_values)


def resolve_config_path(project_root: Path) -> Path:
    """Resolve the active configuration path from EMISSION_CONFIG or the default config name."""
    config_value = os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH)
    return resolve_path(project_root, config_value).resolve()


def load_stage(section_value: dict[str, Any]) -> str:
    """Validate and return the configured workflow stage."""
    unknown_fields = sorted(set(section_value).difference({"stage"}))
    if unknown_fields:
        unknown_text = ", ".join(unknown_fields)
        raise ValueError(f"Unknown keys in config section 'runtime': {unknown_text}")

    stage = str(section_value.get("stage", "all"))
    if stage not in VALID_STAGES:
        valid_text = ", ".join(sorted(VALID_STAGES))
        raise ValueError(f"Invalid stage '{stage}' in config file. Expected one of: {valid_text}")
    return stage


def configure_runtime_resources(config: PipelineConfig) -> Any:
    """Set BLAS/OpenMP/threadpool limits so one Python run respects max_cpu_threads."""
    max_cpu_threads = int(config.max_cpu_threads)
    if max_cpu_threads < 1:
        raise ValueError("pipeline.max_cpu_threads must be at least 1.")

    for env_name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[env_name] = str(max_cpu_threads)

    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError:
        return None
    return threadpool_limits(limits=max_cpu_threads)


def main() -> int:
    """Parse web app CLI arguments and run either self-test or server mode."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = resolve_config_path(project_root)
    config_data = load_yaml_config(config_path)

    runtime_config = ensure_mapping(config_data.get("runtime"), "runtime")
    path_config = ensure_mapping(config_data.get("paths"), "paths")
    pipeline_config = ensure_mapping(config_data.get("pipeline"), "pipeline")

    stage = load_stage(runtime_config)
    paths = build_path_config(project_root, path_config)
    config = build_pipeline_config(pipeline_config)
    threadpool_controller = configure_runtime_resources(config)

    prepared = None
    if stage in {"prepare", "compare", "final", "all"}:
        prepared = prepare_datasets(paths, config)

    if stage in {"compare", "all"}:
        if prepared is None:
            raise RuntimeError("Prepared data is required before model comparison.")
        run_model_comparison(prepared, paths, config)

    if stage in {"final", "all"}:
        if prepared is None:
            raise RuntimeError("Prepared data is required before final model training.")
        model = train_final_model(prepared, paths, config)
        save_training_artifacts(model, prepared.selected_features, paths, config)
        evaluate_model_on_test(model, prepared, paths, config)
        smiles_df, prediction_feature_df, _ = run_prediction_workflow(model, prepared.selected_features, paths, config)
        run_prediction_shap_analysis(
            model,
            smiles_df,
            prediction_feature_df,
            prepared.selected_features,
            paths,
            config,
        )
        if config.auto_cleanup_intermediate_outputs:
            cleanup_intermediate_outputs(paths.output_dir, stage)

    if stage == "predict":
        model, selected_features = load_prediction_artifacts(paths, config)
        smiles_df, prediction_feature_df, _ = run_prediction_workflow(model, selected_features, paths, config)
        run_prediction_shap_analysis(
            model,
            smiles_df,
            prediction_feature_df,
            selected_features,
            paths,
            config,
        )
        if config.auto_cleanup_intermediate_outputs:
            cleanup_intermediate_outputs(paths.output_dir, stage)

    del threadpool_controller

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
