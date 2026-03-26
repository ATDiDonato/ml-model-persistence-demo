from __future__ import annotations

import os
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2 as l2_reg

from src.config import TARGET_COLUMN


def plot_confusion_matrix(cm: np.ndarray) -> None:
    labels = [["TN", "FP"], ["FN", "TP"]]
    annot = [[f"{labels[i][j]}\n{cm[i, j]}" for j in range(2)] for i in range(2)]

    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("Actual Label")
    axes[0].set_title("Confusion Matrix")

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=axes[1],
    )
    axes[1].set_title("Normalised Confusion Matrix (Row-wise)")
    plt.tight_layout()
    plt.show()


def plot_roc_and_pr_curves(
    models: list[dict[str, Any]],
    figsize: tuple[int, int] = (12, 4),
    title_prefix: str = "Model",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    roc_ax, pr_ax = axes

    for i, model in enumerate(models):
        name = model.get("name", f"{title_prefix} {i + 1}")
        if model.get("type") and model.get("stage"):
            name = f"{model['stage']} {model['type']} ({name})"

        y_true = np.asarray(model["y_true"])
        y_score = np.asarray(model["y_score"])

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        roc_ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        pr_ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")

        if i == max(0, len(models) - 1):
            pos_rate = y_true.mean()
            pr_ax.axhline(
                pos_rate,
                linestyle="--",
                label=f"Baseline (Pos rate = {pos_rate:.3f})",
            )

    roc_ax.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC = 0.5)")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_title("ROC Curve")
    roc_ax.legend()

    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Precision-Recall Curve")
    pr_ax.legend()

    plt.tight_layout()
    plt.show()


def compute_performance_metrics(y_true: Any, y_pred: Any, y_prob: Any) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0),
        "auc": roc_auc_score(y_true, y_prob),
    }


def evaluate_and_store_model(
    results: dict[int, dict[str, Any]],
    m_id: int,
    model_name: str,
    model_type: str,
    stage: str,
    y_true: Any,
    y_pred: Any,
    y_prob: Any,
    hyperparameters: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    results[m_id] = {
        "model_name": model_name,
        "model_type": model_type,
        "stage": stage,
        "hyperparameters": hyperparameters or {},
        "metrics": compute_performance_metrics(y_true, y_pred, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "predictions": {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "y_prob": np.asarray(y_prob),
        },
        "metadata": metadata or {},
    }
    return results


def print_model_metrics(results: dict[int, dict[str, Any]], model_id: int) -> None:
    metrics = results[model_id]["metrics"]
    model_type = results[model_id]["model_type"]
    model_name = results[model_id]["model_name"]
    model_stage = results[model_id]["stage"]

    print(f"{model_stage} {model_type} - {model_name} Performance Metrics:")
    print(f"Accuracy:    {float(metrics['accuracy']):.4f}")
    print(f"Precision:   {float(metrics['precision']):.4f}")
    print(f"Recall:      {float(metrics['recall']):.4f}")
    print(f"Specificity: {float(metrics['specificity']):.4f}")
    print(f"AUC:         {float(metrics['auc']):.4f}")


def plot_model_confusion_matrix(results: dict[int, dict[str, Any]], model_id: int) -> None:
    plot_confusion_matrix(results[model_id]["confusion_matrix"])


def build_binary_classifier(
    input_dim: int,
    optimizer: str = "adam",
    units: int = 64,
    layers: int = 1,
    activation: str = "relu",
    dropout: float = 0.2,
    l2_strength: float = 1e-4,
    lr: float = 1e-3,
) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    regularizer = l2_reg(l2_strength) if l2_strength and l2_strength > 0 else None
    for _ in range(layers):
        model.add(Dense(units, activation=activation, kernel_regularizer=regularizer))
        if dropout and dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(1, activation="sigmoid"))

    optimizer_name = optimizer.lower() if isinstance(optimizer, str) else optimizer
    if isinstance(optimizer_name, str):
        if optimizer_name == "adam":
            compiled_optimizer = Adam(learning_rate=lr)
        elif optimizer_name == "rmsprop":
            compiled_optimizer = RMSprop(learning_rate=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    else:
        compiled_optimizer = optimizer_name

    model.compile(
        loss="binary_crossentropy",
        optimizer=compiled_optimizer,
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )
    return model


def choose_n_jobs() -> int:
    cores = os.cpu_count() or 1
    if cores <= 2:
        return cores
    return min(8, int(cores * 0.5))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def create_train_val_test_split_and_scale(
    data_encoded: pd.DataFrame,
    stratify: bool = False,
    seed: int = 42,
    target_col: str = TARGET_COLUMN,
) -> tuple[Any, ...]:
    scaler = StandardScaler().set_output(transform="pandas")

    X = data_encoded.drop(columns=[target_col])
    y = data_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y if stratify else None,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=seed,
        stratify=y_train if stratify else None,
    )

    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return (
        X_train,
        X_train_s,
        X_val,
        X_val_s,
        X_test,
        X_test_s,
        y_train,
        y_val,
        y_test,
        scaler,
    )


def build_models_to_plot(results: dict[int, dict[str, Any]], model_ids: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "name": results[m_id]["model_name"],
            "type": results[m_id]["model_type"],
            "stage": results[m_id]["stage"],
            "y_true": results[m_id]["predictions"]["y_true"],
            "y_score": results[m_id]["predictions"]["y_prob"],
        }
        for m_id in model_ids
    ]
