"""Plot training curves from a Hugging Face Trainer log_history.json.

Run:
python -m src.eval.plot_training outputs/ledgar_legalbert_small/log_history.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--out", default=None, help="output PNG (default: next to the log)")
ap.add_argument("--baseline_acc", type=float, default=0.8835, help="baseline val accuracy")
ap.add_argument("--baseline_f1", type=float, default=0.8185, help="baseline val macro-F1")
args = ap.parse_args()

hist = json.load(open(args.log))
train = [r for r in hist if "loss" in r and "eval_loss" not in r]
seen, evals = set(), []
for r in hist:  # the final evaluation is logged twice; keep one per step
    if "eval_loss" in r and r["step"] not in seen:
        seen.add(r["step"])
        evals.append(r)

ep_tr = [r["epoch"] for r in train]
ep_ev = [r["epoch"] for r in evals]

fig, ax = plt.subplots(2, 2, figsize=(13, 8))

# 1) Loss
ax[0, 0].plot(ep_tr, [r["loss"] for r in train], label="train loss", alpha=0.8)
ax[0, 0].plot(ep_ev, [r["eval_loss"] for r in evals], "o-", label="validation loss")
best = min(evals, key=lambda r: r["eval_loss"])
ax[0, 0].axvline(best["epoch"], ls=":", color="gray", label=f"min val loss (epoch {best['epoch']:.0f})")
ax[0, 0].set_title("Loss")
ax[0, 0].legend()

# 2) Accuracy
ax[0, 1].plot(ep_ev, [r["eval_accuracy"] for r in evals], "o-", label="validation accuracy")
ax[0, 1].axhline(args.baseline_acc, ls="--", color="red", label="TF-IDF baseline (val)")
ax[0, 1].set_title("Validation accuracy")
ax[0, 1].legend()

# 3) Macro-F1
ax[1, 0].plot(ep_ev, [r["eval_macro_f1"] for r in evals], "o-", color="green", label="validation macro-F1")
ax[1, 0].axhline(args.baseline_f1, ls="--", color="red", label="TF-IDF baseline (val)")
ax[1, 0].set_title("Validation macro-F1")
ax[1, 0].legend()

# 4) Learning rate
ax[1, 1].plot(ep_tr, [r["learning_rate"] for r in train], color="purple")
ax[1, 1].set_title("Learning rate")

for a in ax.flat:
    a.set_xlabel("epoch")
    a.grid(alpha=0.3)

fig.tight_layout()
out = Path(args.out) if args.out else Path(args.log).with_name("training_curves.png")
fig.savefig(out, dpi=140)
print(f"Saved {out}")