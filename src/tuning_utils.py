from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def tuner_trials_to_dataframe(tuner: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for trial in tuner.oracle.trials.values():
        row = trial.hyperparameters.values.copy()
        row["trial_id"] = trial.trial_id
        row["status"] = str(trial.status).split(".")[-1]
        for metric_name in [
            "val_auc",
            "val_precision",
            "val_recall",
            "val_accuracy",
            "val_loss",
        ]:
            try:
                row[metric_name] = trial.metrics.get_best_value(metric_name)
            except Exception:
                row[metric_name] = None
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("val_auc", ascending=False)
