"""LEDGAR loader (single-label clause classification, ~100 provision types).

LEDGAR is distributed inside LexGLUE. The Hub copy is stored as Parquet,
so no manual download is needed: the first call downloads and caches it.
"""
from __future__ import annotations

from datasets import DatasetDict, load_dataset

HUB_ID = "coastalcph/lex_glue"
CONFIG = "ledgar"


def load_ledgar(cache_dir: str | None = None) -> DatasetDict:
    """Return a DatasetDict with train / validation / test splits.

    Columns: `text` (str), `label` (ClassLabel, 100 classes).
    """
    try:
        return load_dataset(HUB_ID, CONFIG, cache_dir=cache_dir)
    except Exception as err:  # pragma: no cover - network/version dependent
        print(f"[ledgar] default load failed ({err!r}); trying Parquet files directly")
        base = f"hf://datasets/{HUB_ID}/{CONFIG}"
        files = {s: f"{base}/{s}/0000.parquet" for s in ("train", "validation", "test")}
        return load_dataset("parquet", data_files=files, cache_dir=cache_dir)


def get_label_names(ds: DatasetDict) -> list[str]:
    """Human-readable label names, ordered by label id."""
    return list(ds["train"].features["label"].names)
