"""Most common confusions from a test_predictions.npz file.

Run:  python -m src.eval.confusions outputs/ledgar_legalbert_small/test_predictions.npz
"""
import sys
from collections import Counter

import numpy as np

data = np.load(sys.argv[1])
y_true, y_pred, names = data["y_true"], data["y_pred"], data["names"]
pairs = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
print(f"{'true label':<30} {'predicted as':<30} count")
for (t, p), n in pairs.most_common(25):
    print(f"{names[t]:<30} {names[p]:<30} {n}")