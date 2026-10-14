"""TF-IDF + linear baselines on LEDGAR.

Models are compared on the validation split; only the best one is scored
on the test split (so the test set is not used for model selection).

Run:  python -m src.train.baseline_ledgar --config configs/ledgar_baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.ledgar import get_label_names, load_ledgar
from src.eval.metrics import compute_metrics, per_class_report


def build_pipeline(name: str, params: dict, tfidf_cfg: dict, seed: int) -> Pipeline:
    vec = TfidfVectorizer(
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        min_df=tfidf_cfg["min_df"],
        max_features=tfidf_cfg["max_features"],
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        lowercase=tfidf_cfg["lowercase"],
    )
    if name.startswith("linear_svc"):
        clf = LinearSVC(
            C=params.get("C", 1.0),
            class_weight=params.get("class_weight"),
            random_state=seed,
        )
    elif name.startswith("logreg"):
        clf = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 300),
            class_weight=params.get("class_weight"),
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model type: {name}")
    return Pipeline([("tfidf", vec), ("clf", clf)])


def main(config_path: str) -> None:
    cfg = yaml.safe_load(open(config_path))
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    ds = load_ledgar()
    names = get_label_names(ds)
    X_tr, y_tr = ds["train"]["text"], np.array(ds["train"]["label"])
    X_va, y_va = ds["validation"]["text"], np.array(ds["validation"]["label"])
    X_te, y_te = ds["test"]["text"], np.array(ds["test"]["label"])
    print(f"train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}  labels={len(names)}")

    results, fitted = {}, {}
    for name, params in cfg["models"].items():
        print(f"\n== {name} ==")
        t0 = time.time()
        pipe = build_pipeline(name, params, cfg["tfidf"], seed)
        pipe.fit(X_tr, y_tr)
        val = compute_metrics(y_va, pipe.predict(X_va))
        val["fit_seconds"] = round(time.time() - t0, 1)
        results[name] = {"params": params, "val": val}
        fitted[name] = pipe
        print(val)

    best = max(results, key=lambda n: results[n]["val"]["macro_f1"])
    print(f"\nBest on validation (macro-F1): {best}")

    test_pred = fitted[best].predict(X_te)
    test = compute_metrics(y_te, test_pred)
    print("TEST:", test)

    report = per_class_report(y_te, test_pred, names)
    report.to_csv(out / "per_class_test.csv", index=False)
    print("\nWorst 10 labels on test:\n", report.head(10).to_string(index=False))

    summary = {
        "seed": seed,
        "best_model": best,
        "validation": {n: r["val"] for n, r in results.items()},
        "test": test,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    joblib.dump({"pipeline": fitted[best], "label_names": names}, out / "best_model.joblib")
    np.savez(out / "test_predictions.npz", y_true=y_te, y_pred=test_pred, names=np.array(names))
    print(f"\nSaved metrics.json, per_class_test.csv, best_model.joblib to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ledgar_baseline.yaml")
    main(ap.parse_args().config)
