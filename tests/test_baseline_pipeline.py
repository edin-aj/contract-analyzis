"""Offline check that the baseline pipeline fits and predicts on toy data."""
import yaml

from src.train.baseline_ledgar import build_pipeline


def test_pipeline_fits_toy_data():
    cfg = yaml.safe_load(open("configs/ledgar_baseline.yaml"))
    texts = ["governing law of new york", "this agreement is governed by delaware law",
             "payment due within 30 days", "invoices are payable in thirty days"] * 5
    labels = [0, 0, 1, 1] * 5
    for name, params in cfg["models"].items():
        pipe = build_pipeline(name, params, {**cfg["tfidf"], "min_df": 1}, seed=0)
        pipe.fit(texts, labels)
        assert len(pipe.predict(texts)) == len(texts)
