"""Quick look at LEDGAR: sizes, label imbalance, text lengths.

Run:  python -m src.data.explore_ledgar
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.ledgar import get_label_names, load_ledgar


def main(out_dir: str = "outputs/ledgar_explore") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_ledgar()
    names = get_label_names(ds)
    print(ds)
    print(f"\n{len(names)} labels")
    for split in ds:
        print(f"{split:>10}: {len(ds[split]):,} examples")

    train = ds["train"].to_pandas()
    train["label_name"] = train["label"].map(lambda i: names[i])
    train["n_words"] = train["text"].str.split().str.len()

    counts = train["label_name"].value_counts()
    print("\nMost common labels:\n", counts.head(10).to_string())
    print("\nRarest labels:\n", counts.tail(10).to_string())
    print(f"\nImbalance ratio (max/min class): {counts.max() / counts.min():.1f}")

    q = train["n_words"].quantile([0.5, 0.9, 0.95, 0.99, 1.0])
    print("\nWords per provision (quantiles):\n", q.to_string())
    print("\nTip: roughly 1.3 tokens per word; compare against your max_len.")

    counts.to_csv(out / "label_counts.csv")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(range(len(counts)), counts.values)
    ax[0].set_title("Training examples per label (sorted)")
    ax[0].set_xlabel("label rank")
    ax[1].hist(train["n_words"].clip(upper=600), bins=60)
    ax[1].set_title("Words per provision (clipped at 600)")
    fig.tight_layout()
    fig.savefig(out / "ledgar_overview.png", dpi=130)
    print(f"\nSaved plots/tables to {out}/")

    print("\nSample provisions:")
    for _, row in train.sample(3, random_state=0).iterrows():
        print(f"\n[{row['label_name']}]\n{row['text'][:400]}")


if __name__ == "__main__":
    main()
