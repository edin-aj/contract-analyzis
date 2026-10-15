"""Compare two models on the LEDGAR test set.

Run:
python -m src.eval.compare outputs/ledgar_baseline/test_predictions.npz outputs/ledgar_legalbert_small/test_predictions.npz
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import compute_metrics, per_class_report


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["y_true"], d["y_pred"], list(d["names"])


def top_confusions(y_true, y_pred, names, k=10):
    c = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    return [(names[t], names[p], n) for (t, p), n in c.most_common(k)]


base_path, model_path = sys.argv[1], sys.argv[2]
y_true, y_base, names = load(base_path)
y_true2, y_model, names2 = load(model_path)
assert (y_true == y_true2).all() and names == names2, "Different test sets or label order!"

# 1) Overall metrics
rows = {"baseline": compute_metrics(y_true, y_base), "transformer": compute_metrics(y_true, y_model)}
overall = pd.DataFrame(rows).T.round(4)
print("\n=== Overall (test) ===\n", overall.to_string())

# 2) Per-label F1 difference
rb = per_class_report(y_true, y_base, names).set_index("label")
rm = per_class_report(y_true, y_model, names).set_index("label")
diff = pd.DataFrame({
    "f1_baseline": rb["f1"], "f1_transformer": rm["f1"],
    "delta": rm["f1"] - rb["f1"], "support": rb["support"],
}).sort_values("delta")
print("\n=== Labels where the transformer is WORSE ===\n", diff.head(10).round(3).to_string())
print("\n=== Labels where the transformer is BETTER ===\n", diff.tail(10).round(3).to_string())

# 3) Where do they disagree?
b_ok, m_ok = y_base == y_true, y_model == y_true
print("\n=== Agreement ===")
print(f"both correct       : {(b_ok & m_ok).sum()}")
print(f"only baseline right: {(b_ok & ~m_ok).sum()}")
print(f"only model right   : {(~b_ok & m_ok).sum()}")
print(f"both wrong         : {(~b_ok & ~m_ok).sum()}")

# 4) Top confusions per model
for tag, y in (("baseline", y_base), ("transformer", y_model)):
    print(f"\n=== Top confusions: {tag} (true -> predicted, count) ===")
    for t, p, n in top_confusions(y_true, y, names):
        print(f"{t:<28} -> {p:<28} {n}")

out = Path("outputs/comparison")
out.mkdir(parents=True, exist_ok=True)
overall.to_csv(out / "overall.csv")
diff.to_csv(out / "per_label_delta.csv")
print(f"\nSaved tables to {out}/")