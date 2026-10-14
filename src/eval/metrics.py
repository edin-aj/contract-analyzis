"""Shared classification metrics used by baselines and transformer runs."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def per_class_report(y_true, y_pred, label_names: list[str]) -> pd.DataFrame:
    """Per-label precision/recall/F1/support, sorted by F1 (worst first)."""
    labels = list(range(len(label_names)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    df = pd.DataFrame(
        {"label": label_names, "precision": p, "recall": r, "f1": f, "support": s}
    )
    return df.sort_values("f1").reset_index(drop=True)
