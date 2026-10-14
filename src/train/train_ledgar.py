"""Fine-tune a (Legal-)BERT model on LEDGAR.

Run:  python -m src.train.train_ledgar --config configs/ledgar_smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data.ledgar import get_label_names, load_ledgar
from src.eval.metrics import compute_metrics, per_class_report


def main(config_path: str) -> None:
    cfg = yaml.safe_load(open(config_path))
    seed = cfg["seed"]
    set_seed(seed)

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    ds = load_ledgar()
    names = get_label_names(ds)

    # Optional subsampling (used by the smoke test)
    if cfg.get("max_train_samples"):
        n = min(cfg["max_train_samples"], len(ds["train"]))
        ds["train"] = ds["train"].shuffle(seed=seed).select(range(n))
    if cfg.get("max_eval_samples"):
        for split in ("validation", "test"):
            n = min(cfg["max_eval_samples"], len(ds[split]))
            ds[split] = ds[split].select(range(n))

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])

    def encode(batch):
        return tok(batch["text"], truncation=True, max_length=cfg["max_length"])

    ds = ds.map(encode, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=len(names),
        id2label={i: n for i, n in enumerate(names)},
        label2id={n: i for i, n in enumerate(names)},
    )

    def metrics_fn(p):
        return compute_metrics(p.label_ids, np.argmax(p.predictions, axis=-1))

    eff_bs = cfg["batch_size"] * cfg["grad_accum"]
    total_steps = math.ceil(len(ds["train"]) / eff_bs) * cfg["epochs"]
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        fp16=bool(cfg["fp16"]) and torch.cuda.is_available(),
        gradient_checkpointing=bool(cfg["gradient_checkpointing"]),
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics_fn,
    )

    trainer.train()

    val = {k.replace("eval_", ""): v for k, v in trainer.evaluate(ds["validation"]).items()}
    pred = trainer.predict(ds["test"])
    y_pred = pred.predictions.argmax(-1)
    y_true = pred.label_ids
    test = compute_metrics(y_true, y_pred)
    print("VAL :", val)
    print("TEST:", test)

    per_class_report(y_true, y_pred, names).to_csv(out / "per_class_test.csv", index=False)
    np.savez(out / "test_predictions.npz", y_true=y_true, y_pred=y_pred, names=np.array(names))
    (out / "metrics.json").write_text(
        json.dumps({"config": cfg, "validation": val, "test": test}, indent=2, default=float)
    )
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    trainer.save_model(str(out / "best_model"))
    tok.save_pretrained(str(out / "best_model"))
    print(f"Saved results and best_model to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)