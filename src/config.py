from __future__ import annotations

DEFAULT_RANDOM_SEED = 42
TARGET_COLUMN = "dropout"

STAGE_DATA_FILENAMES = {
    "stage_1": "Stage_1_public.csv",
}

TUNING_ARTIFACT_FILENAMES = {
    "best_params": "best_params.json",
    "trials": "trials.csv",
    "search_summary": "search_summary.txt",
    "best_model_training": "best_model_training.keras",
    "optuna_study": "optuna_study.sqlite3",
    "best_model_joblib": "best_model.joblib",
}
