#!/usr/bin/env python3
"""
Download the public Sm-BFO composition-series STEM dataset.

Official sources (same file AtomAI ``stem_smbfo`` retrieves):
  - Direct file: https://zenodo.org/record/4876786/files/composition_series_dict_full.npy
  - Dataset DOI: https://doi.org/10.13139/ORNLNCCS/1773704
  - Paper:       https://doi.org/10.1038/s41524-020-00396-2

Saves to ``data/SmBFO_composition_series.npy`` (pickled dict: keys → main_image, xy_COM, …).

Example:
  python scripts/download_sm_bfo_dataset.py
  python scripts/download_sm_bfo_dataset.py --out data/SmBFO_composition_series.npy
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ZENODO_FILE_URL = (
    "https://zenodo.org/record/4876786/files/composition_series_dict_full.npy"
)
DATASET_DOI = "https://doi.org/10.13139/ORNLNCCS/1773704"
PAPER_DOI = "https://doi.org/10.1038/s41524-020-00396-2"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading\n  {url}\n→ {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e9:.2f} GB)")


def main() -> None:
    p = argparse.ArgumentParser(description="Download Sm-BFO composition series (.npy)")
    p.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "data" / "SmBFO_composition_series.npy"),
        help="Output path for the raw dict .npy",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Optional JSON manifest path (default: <out_dir>/dataset_manifest.json)",
    )
    args = p.parse_args()

    out = Path(args.out)
    manifest_path = Path(args.manifest) if args.manifest else out.parent / "dataset_manifest.json"

    download(ZENODO_FILE_URL, out)

    manifest = {
        "filename": out.name,
        "path": str(out.resolve()),
        "zenodo_record": "https://zenodo.org/record/4876786",
        "zenodo_file_url": ZENODO_FILE_URL,
        "dataset_doi": DATASET_DOI,
        "paper_doi": PAPER_DOI,
        "format": "numpy pickled dict[str, dict]; keys are composition ids (e.g. Sm_7_0)",
        "fields_per_entry": ["main_image", "xy_COM", "Pxy", "..."],
        "preprocess_command": (
            "python -m gan_seg.preprocess_dataset "
            f"--source {out} --out data/processed/sm_bfo_com"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"If you use this dataset, please cite: {DATASET_DOI}")


if __name__ == "__main__":
    main()
