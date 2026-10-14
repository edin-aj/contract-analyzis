import numpy as np

from src.eval.metrics import compute_metrics, per_class_report


def test_perfect_predictions():
    y = np.array([0, 1, 2, 1, 0])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0


def test_per_class_report_includes_absent_labels():
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 1, 1])
    df = per_class_report(y_true, y_pred, ["a", "b", "c"])
    assert set(df["label"]) == {"a", "b", "c"}
    assert df.loc[df["label"] == "c", "support"].iloc[0] == 0
