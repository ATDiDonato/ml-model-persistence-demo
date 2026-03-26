from __future__ import annotations

import os
import sys
from pathlib import Path

from src.config import STAGE_DATA_FILENAMES


REPO_MARKERS = ("src", "data", "notebooks")
DEFAULT_COLAB_REPO_NAME = "ml-model-persistence-demo"
DEFAULT_COLAB_REPO_PARENT = Path("/content")


def is_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def _is_project_root(candidate: Path) -> bool:
    return all((candidate / marker).exists() for marker in REPO_MARKERS)


def resolve_project_root(start_path: str | Path | None = None) -> Path:
    search_roots: list[Path] = []

    if start_path is not None:
        search_roots.append(Path(start_path).resolve())
    else:
        current_file = Path(__file__).resolve()
        search_roots.extend(
            [
                Path.cwd().resolve(),
                current_file.parent,
                current_file.parent.parent,
            ]
        )

    colab_repo_name = os.environ.get("COLAB_PROJECT_REPO", DEFAULT_COLAB_REPO_NAME)
    colab_repo_root = DEFAULT_COLAB_REPO_PARENT / colab_repo_name
    if is_colab():
        search_roots.extend([colab_repo_root, DEFAULT_COLAB_REPO_PARENT])

    seen: set[Path] = set()
    for root in search_roots:
        for candidate in [root, *root.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if _is_project_root(candidate):
                return candidate

    raise FileNotFoundError(
        "Could not locate the repository root. In Colab, clone the repo into "
        f"{colab_repo_root} or set COLAB_PROJECT_REPO to the cloned folder name."
    )


def ensure_directories(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


PROJECT_ROOT = resolve_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def get_stage_data_path(stage_name: str) -> Path:
    return DATA_DIR / STAGE_DATA_FILENAMES[stage_name]
