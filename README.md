# ML Model Persistence Demo

This repo is a standalone demo for save/load workflows around datasets, trained models, and tuning artefacts. The main entry point is [notebooks/demo_model_and_artifact_persistence.ipynb](/home/alextd/projects/ml-model-persistence-demo/notebooks/demo_model_and_artifact_persistence.ipynb).

The notebook demonstrates:

- saving and reloading a CSV dataset copy
- saving and loading a baseline XGBoost model with `joblib`
- saving and loading a baseline Keras model plus training history
- saving, loading, resuming, and overwriting Keras Tuner artefacts
- a manual neural-network for-loop grid search with saved models, histories, CSV, and JSON summaries
- an optional `GridSearchCV` example for XGBoost
- repo-relative artifact handling that also works in Colab and can be paired with Google Drive

## Repo layout

The demo keeps all generated outputs under `demo_artifacts/`:

```text
demo_artifacts/
  datasets/
  models/
    xgboost/
    neural_network/
  tuning/
    gridsearch/
    xgboost/
    keras_tuner/
    nn_grid_search/
      histories/
      models/
  figures/
```

The source dataset used by the notebook lives at `data/processed/Stage_1_public.csv`. The notebook also writes a demo copy into `demo_artifacts/datasets/` to make the CSV persistence step explicit.

## Running locally

1. Create and activate a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Open [notebooks/demo_model_and_artifact_persistence.ipynb](/home/alextd/projects/ml-model-persistence-demo/notebooks/demo_model_and_artifact_persistence.ipynb) and run it from the cloned repo.

The notebook uses repo-relative paths, so local runs and reruns will keep loading from and saving to the same `demo_artifacts/` tree.

## Running in Colab

Clone the repo into `/content/ml-model-persistence-demo` and install dependencies:

```bash
!git clone <repo-url> /content/ml-model-persistence-demo
%cd /content/ml-model-persistence-demo
!pip install -r requirements.txt
```

If the cloned folder name is different, set:

```python
import os
os.environ["COLAB_PROJECT_REPO"] = "<your-cloned-folder-name>"
```

before running the notebook bootstrap cell.

Optional Google Drive support:

- mount Drive if you want persistence across Colab runtime resets
- either clone the repo inside Drive or copy `demo_artifacts/` into Drive after a run
- the notebook itself stays repo-relative, so no code changes are required just to use Drive-backed storage

## Persistence behaviour

The notebook uses one precedence rule throughout:

`OVERWRITE` > `LOAD` > create/train/tune fresh

That behaviour is shown for:

- baseline model artefacts
- saved `GridSearchCV` summaries
- Optuna-based XGBoost tuning artefacts
- Keras Tuner project state and exported best model
- manual NN grid-search results and saved histories
