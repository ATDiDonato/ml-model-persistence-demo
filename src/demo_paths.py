from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.paths import PROJECT_ROOT, ensure_directories


DEFAULT_DEMO_ARTIFACT_DIRNAME = "demo_artifacts"
DEMO_ARTIFACT_ROOT_ENV_VAR = "DEMO_ARTIFACT_ROOT"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / DEFAULT_DEMO_ARTIFACT_DIRNAME


def resolve_artifact_root() -> Path:
    artifact_root_override = os.environ.get(DEMO_ARTIFACT_ROOT_ENV_VAR)
    if artifact_root_override:
        return Path(artifact_root_override).expanduser()
    return DEFAULT_ARTIFACT_ROOT


@dataclass(frozen=True)
class DemoPaths:
    project_root: Path
    artifact_root: Path

    @property
    def artifact_reference_root(self) -> Path:
        return self.artifact_root.parent

    def to_artifact_reference(self, path: str | Path) -> str:
        artifact_path = Path(path)
        if not artifact_path.is_absolute():
            artifact_path = self.artifact_reference_root / artifact_path

        try:
            return artifact_path.relative_to(self.artifact_reference_root).as_posix()
        except ValueError:
            return artifact_path.as_posix()

    def resolve_artifact_reference(self, reference: str | Path) -> Path:
        artifact_reference = Path(reference)
        if artifact_reference.is_absolute():
            return artifact_reference
        return self.artifact_reference_root / artifact_reference


DEMO_PATHS = DemoPaths(
    project_root=PROJECT_ROOT,
    artifact_root=resolve_artifact_root(),
)

ARTIFACT_ROOT = DEMO_PATHS.artifact_root
DEMO_ARTIFACT_DIR = ARTIFACT_ROOT

DEMO_DATASET_DIR = DEMO_ARTIFACT_DIR / "datasets"

DEMO_MODEL_DIR = DEMO_ARTIFACT_DIR / "models"
DEMO_XGB_MODEL_DIR = DEMO_MODEL_DIR / "xgboost"
DEMO_NN_MODEL_DIR = DEMO_MODEL_DIR / "neural_network"

DEMO_TUNING_DIR = DEMO_ARTIFACT_DIR / "tuning"
DEMO_GRIDSEARCH_DIR = DEMO_TUNING_DIR / "gridsearch"
DEMO_XGB_TUNING_DIR = DEMO_TUNING_DIR / "xgboost"
DEMO_KERAS_TUNER_DIR = DEMO_TUNING_DIR / "keras_tuner"
DEMO_NN_GRID_SEARCH_DIR = DEMO_TUNING_DIR / "nn_grid_search"
DEMO_NN_GRID_SEARCH_HISTORY_DIR = DEMO_NN_GRID_SEARCH_DIR / "histories"
DEMO_NN_GRID_SEARCH_MODEL_DIR = DEMO_NN_GRID_SEARCH_DIR / "models"

DEMO_FIGURES_DIR = DEMO_ARTIFACT_DIR / "figures"
DEMO_MPLCONFIG_DIR = DEMO_ARTIFACT_DIR / ".mplconfig"


def to_artifact_reference(path: str | Path) -> str:
    return DEMO_PATHS.to_artifact_reference(path)


def resolve_artifact_reference(reference: str | Path) -> Path:
    return DEMO_PATHS.resolve_artifact_reference(reference)


def ensure_demo_directories() -> None:
    ensure_directories(
        DEMO_ARTIFACT_DIR,
        DEMO_DATASET_DIR,
        DEMO_MODEL_DIR,
        DEMO_XGB_MODEL_DIR,
        DEMO_NN_MODEL_DIR,
        DEMO_TUNING_DIR,
        DEMO_GRIDSEARCH_DIR,
        DEMO_XGB_TUNING_DIR,
        DEMO_KERAS_TUNER_DIR,
        DEMO_NN_GRID_SEARCH_DIR,
        DEMO_NN_GRID_SEARCH_HISTORY_DIR,
        DEMO_NN_GRID_SEARCH_MODEL_DIR,
        DEMO_FIGURES_DIR,
        DEMO_MPLCONFIG_DIR,
    )
