"""Re-create test predictions from the saved baseline model (no retraining).

Run:  python -m src.eval.baseline_predictions
"""
import joblib
import numpy as np

from src.data.ledgar import load_ledgar

saved = joblib.load("outputs/ledgar_baseline/best_model.joblib")
pipe, names = saved["pipeline"], saved["label_names"]

ds = load_ledgar()
y_true = np.array(ds["test"]["label"])
y_pred = pipe.predict(ds["test"]["text"])

np.savez("outputs/ledgar_baseline/test_predictions.npz",
         y_true=y_true, y_pred=y_pred, names=np.array(names))
print("accuracy:", (y_true == y_pred).mean())