"""Shared rules for manuscript figure exports: real data only unless explicitly opted in."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PREPROCESS_HINT = (
    "  python -m gan_seg.preprocess_dataset --source data/SmBFO_composition_series.npy "
    "--out data/processed/sm_bfo_com\n"
    "(Requires `data/SmBFO_composition_series.npy` and produces val/*.npz shards.)"
)

SYNTHETIC_FLAG = "--allow-synthetic-demo-only"


def exit_missing_processed_val(processed: str) -> None:
    root = Path(processed)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        print(
            f"ERROR: Missing manuscript-grade data at {root.resolve()}.\n"
            "Journal figures must use real Sm-BFO preprocessed val patches, not placeholders.\n"
            + PREPROCESS_HINT
            + f"\nInternal layout testing only (never for submission): pass {SYNTHETIC_FLAG}\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        man = json.loads(manifest.read_text(encoding="utf-8"))
        val_shards = man.get("splits", {}).get("val", {}).get("shards", [])
    except (OSError, json.JSONDecodeError, KeyError):
        val_shards = []
    if not val_shards:
        print(
            f"ERROR: {root.resolve()} has no val shards in manifest.\n"
            + PREPROCESS_HINT
            + f"\nInternal layout testing only: {SYNTHETIC_FLAG}\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
    for rel in val_shards[:1]:
        if not (root / rel).is_file():
            print(
                f"ERROR: Val shard missing on disk: {root / rel}\n"
                "Re-run preprocessing or restore data before exporting figures.\n"
                + f"Internal layout testing only: {SYNTHETIC_FLAG}\n",
                file=sys.stderr,
            )
            raise SystemExit(1)
