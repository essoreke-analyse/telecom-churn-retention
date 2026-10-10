"""Train the final models on all data and save them."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import sklearn

from churn.compare import SUSPECT_FEATURES
from churn.data import PROJECT_ROOT, load_config, load_raw
from churn.explain import BOOSTING_PARAMS
from churn.features import build_features, split_xy
from churn.tune import base_pipeline

MODEL_DIR = PROJECT_ROOT / "models"
VERSIONS = ("full", "early_warning")


def select_features(X: pd.DataFrame, version: str) -> pd.DataFrame:
    """'early_warning' drops the two variables that may be consequences of leaving."""
    if version == "early_warning":
        return X.drop(columns=SUSPECT_FEATURES)
    if version == "full":
        return X
    raise ValueError(f"Unknown version: {version}")


def fit_final(X: pd.DataFrame, y, version: str, seed: int = 42) -> dict:
    """Fit the retained gradient boosting on all rows; return a model bundle."""
    X = select_features(X, version)
    pipe = base_pipeline("hist_gradient_boosting", X.columns, seed)
    pipe.set_params(**BOOSTING_PARAMS).fit(X, y)
    return {
        "pipeline": pipe,
        "features": list(X.columns),
        "version": version,
        "n_train": int(len(X)),
        "train_churn_rate": float(pd.Series(y).mean()),
        "sklearn_version": sklearn.__version__,
    }


def model_path(version: str, model_dir: Path | None = None) -> Path:
    return (model_dir or MODEL_DIR) / f"churn_{version}.joblib"


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(version: str, model_dir: Path | None = None) -> dict:
    path = model_path(version, model_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m churn.final")
    return joblib.load(path)


def main() -> None:
    seed = load_config()["project"]["seed"]
    X, y = split_xy(build_features(load_raw()))
    for version in VERSIONS:
        bundle = fit_final(X, y, version, seed)
        path = model_path(version)
        save_bundle(bundle, path)
        print(f"Saved {version} model ({len(bundle['features'])} features, "
              f"{bundle['n_train']} rows) -> {path}")


if __name__ == "__main__":
    main()