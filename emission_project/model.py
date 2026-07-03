from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:
    CatBoostRegressor = None


FINAL_XGB_BASE_PARAMS = {
    "n_estimators": 600,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 1.0,
    "objective": "reg:squarederror",
    "min_child_weight": 1,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
}


class MeanRegressorEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, estimators: list[tuple[str, Any]] | None = None) -> None:
        self.estimators = estimators

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MeanRegressorEnsemble":
        if not self.estimators:
            raise ValueError("MeanRegressorEnsemble requires at least one base estimator.")

        self.estimators_ = []
        for name, estimator in self.estimators:
            fitted_estimator = clone(estimator)
            fitted_estimator.fit(x, y)
            self.estimators_.append((name, fitted_estimator))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimators_") or not self.estimators_:
            raise ValueError("MeanRegressorEnsemble must be fitted before prediction.")
        predictions = [estimator.predict(x) for _, estimator in self.estimators_]
        return np.mean(np.column_stack(predictions), axis=1)

    @property
    def named_estimators_(self) -> dict[str, Any]:
        if not hasattr(self, "estimators_"):
            return {}
        return dict(self.estimators_)


def build_model_specs(random_state: int, max_cpu_threads: int) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    specs: dict[str, tuple[Any, dict[str, list[Any]]]] = {}

    if CatBoostRegressor is not None:
        specs["CatBoost"] = (
            CatBoostRegressor(
                loss_function="RMSE",
                random_seed=random_state,
                verbose=False,
                thread_count=max_cpu_threads,
                allow_writing_files=False,
            ),
            {
                "iterations": [300, 600],
                "depth": [4, 6],
                "learning_rate": [0.03, 0.08],
                "l2_leaf_reg": [3.0, 5.0],
            },
        )

    specs.update(
        {
            "XGBoost": (
                XGBRegressor(
                    random_state=random_state,
                    n_jobs=max_cpu_threads,
                    verbosity=0,
                    objective="reg:squarederror",
                    tree_method="hist",
                    device="cpu",
                ),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1],
                    "subsample": [0.8, 1.0],
                },
            ),
            "LightGBM": (
                LGBMRegressor(
                    random_state=random_state,
                    n_jobs=max_cpu_threads,
                    verbosity=-1,
                    device_type="cpu",
                ),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1],
                    "num_leaves": [31, 63],
                },
            ),
            "RF": (
                RandomForestRegressor(random_state=random_state, n_jobs=max_cpu_threads),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 8, 12],
                    "min_samples_split": [2, 5],
                },
            ),
            "GBR": (
                GradientBoostingRegressor(random_state=random_state),
                {
                    "n_estimators": [200, 500],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "max_depth": [2, 3],
                    "subsample": [0.8, 1.0],
                },
            ),
            "KNN": (
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", KNeighborsRegressor()),
                    ]
                ),
                {
                    "model__n_neighbors": [3, 5, 10],
                    "model__weights": ["uniform", "distance"],
                    "model__p": [1, 2],
                },
            ),
            "KRR": (
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", KernelRidge()),
                    ]
                ),
                {
                    "model__alpha": [0.001, 0.01, 0.1],
                    "model__gamma": [0.001, 0.01, 0.1],
                    "model__kernel": ["rbf"],
                },
            ),
            "SVR": (
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", SVR()),
                    ]
                ),
                {
                    "model__C": [1, 10, 100],
                    "model__gamma": [0.001, 0.01, 0.1],
                    "model__epsilon": [0.01, 0.1],
                },
            ),
        }
    )
    return specs


def build_final_xgb_param_distributions() -> dict[str, list[Any]]:
    return {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0.0, 0.1, 0.3],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [1.0, 3.0, 5.0],
    }


def build_final_xgb(
    random_state: int,
    max_cpu_threads: int,
    param_overrides: dict[str, Any] | None = None,
) -> XGBRegressor:
    params = dict(FINAL_XGB_BASE_PARAMS)
    if param_overrides:
        params.update(param_overrides)

    return XGBRegressor(
        **params,
        random_state=random_state,
        n_jobs=max_cpu_threads,
        verbosity=0,
        tree_method="hist",
        device="cpu",
    )


def build_final_lgbm(random_state: int, max_cpu_threads: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=max_cpu_threads,
        verbosity=-1,
    )


def build_final_cat(random_state: int, max_cpu_threads: int) -> Any:
    if CatBoostRegressor is None:
        raise ModuleNotFoundError("CatBoost is required for the CAT base model.")
    return CatBoostRegressor(
        iterations=600,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False,
        thread_count=max_cpu_threads,
        allow_writing_files=False,
    )


def build_final_rf(random_state: int, max_cpu_threads: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=random_state,
        n_jobs=max_cpu_threads,
    )


def build_final_mean_ensemble(
    algorithm_names: list[str],
    random_state: int,
    max_cpu_threads: int,
    xgb_param_overrides: dict[str, Any] | None = None,
) -> MeanRegressorEnsemble:
    builders = {
        "XGB": lambda: build_final_xgb(random_state, max_cpu_threads, xgb_param_overrides),
        "CAT": lambda: build_final_cat(random_state, max_cpu_threads),
        "LGBM": lambda: build_final_lgbm(random_state, max_cpu_threads),
        "RF": lambda: build_final_rf(random_state, max_cpu_threads),
    }
    estimators: list[tuple[str, Any]] = []
    for algorithm_name in algorithm_names:
        normalized_name = str(algorithm_name).strip().upper()
        if normalized_name == "LGB":
            normalized_name = "LGBM"
        if normalized_name not in builders:
            valid = ", ".join(sorted(builders))
            raise ValueError(f"Unsupported ensemble algorithm '{algorithm_name}'. Expected one of: {valid}.")
        estimators.append((normalized_name, builders[normalized_name]()))
    return MeanRegressorEnsemble(estimators=estimators)


def get_model_artifact_type(model: RegressorMixin) -> str:
    if isinstance(model, XGBRegressor):
        return "xgb"
    if isinstance(model, MeanRegressorEnsemble):
        return "mean_ensemble"
    return type(model).__name__


def save_final_model(model: RegressorMixin, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, XGBRegressor):
        model.save_model(model_path)
        return
    joblib.dump(model, model_path)


def load_final_model(model_path: Path) -> RegressorMixin:
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    if model_path.suffix.lower() == ".json":
        return load_final_xgb(model_path)
    return joblib.load(model_path)


def get_model_display_name(model: RegressorMixin) -> str:
    if isinstance(model, XGBRegressor):
        return "XGBoost"
    if isinstance(model, MeanRegressorEnsemble):
        model_names = "+".join(model.named_estimators_.keys())
        return f"{model_names} Mean Ensemble" if model_names else "Mean Ensemble"
    return type(model).__name__


def resolve_explainability_model(model: RegressorMixin) -> tuple[XGBRegressor | None, str]:
    if isinstance(model, XGBRegressor):
        return model, "final_model"
    if isinstance(model, MeanRegressorEnsemble):
        xgb_model = model.named_estimators_.get("XGB")
        if isinstance(xgb_model, XGBRegressor):
            return xgb_model, "xgb_base_proxy"
    return None, "unsupported"


def load_final_xgb(model_path: Path) -> XGBRegressor:
    if not model_path.exists():
        raise FileNotFoundError(f"Saved XGBoost model not found: {model_path}")

    model = XGBRegressor()
    model.load_model(model_path)
    return model
