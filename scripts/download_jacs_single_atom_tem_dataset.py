#!/usr/bin/env python3
"""
Download: "Dataset for Automated Image Analysis for Single-Atom Detection in
Catalytic Materials by Transmission Electron Microscopy"

- Paper: JACS, Mitchell et al., https://doi.org/10.1021/jacs.1c12466
- Data: Zenodo https://doi.org/10.5281/zenodo.5931544
- Code (SAC-CNN): https://github.com/HPAI-BSC/AtomDetection_ACSTEM

Fetches Readme.txt + Supplementary_data_ja-2021-12466d.zip and extracts the zip.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "external_stem_data" / "jacs_single_atom_TEM"

ZENODO_RECORD = "5931544"
ZENODO_FILES_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"
FILES = {
    "Readme.txt": f"https://zenodo.org/record/{ZENODO_RECORD}/files/Readme.txt?download=1",
    "Supplementary_data_ja-2021-12466d.zip": f"https://zenodo.org/record/{ZENODO_RECORD}/files/Supplementary_data_ja-2021-12466d.zip?download=1",
}


def _download(url: str, dest: Path, reporthook=None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest), reporthook=reporthook)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory for downloads and extracted data",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Only extract if zip already present",
    )
    p.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download only, do not unzip",
    )
    args = p.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "title": "Dataset for Automated Image Analysis for Single-Atom Detection in Catalytic Materials by Transmission Electron Microscopy",
        "paper_doi": "https://doi.org/10.1021/jacs.1c12466",
        "dataset_doi": "https://doi.org/10.5281/zenodo.5931544",
        "zenodo_record": ZENODO_RECORD,
        "zenodo_files_api": ZENODO_FILES_API,
        "github": "https://github.com/HPAI-BSC/AtomDetection_ACSTEM",
        "license": "CC-BY-4.0",
        "out_dir": str(out),
    }

    if not args.skip_download:
        for name, url in FILES.items():
            target = out / name
            if target.is_file() and target.stat().st_size > 0:
                print(f"[skip exists] {target}")
                continue
            print(f"[download] {name} -> {target}")

            def hook(block, bs, total):
                if total <= 0:
                    return
                done = min(block * bs, total)
                pct = 100.0 * done / total
                sys.stdout.write(f"\r  {pct:5.1f}% ({done} / {total} bytes)")
                sys.stdout.flush()

            _download(url, target, reporthook=hook if name.endswith(".zip") else None)
            if name.endswith(".zip"):
                print()

    readme = out / "Readme.txt"
    zpath = out / "Supplementary_data_ja-2021-12466d.zip"
    if not zpath.is_file():
        print("Missing zip; run without --skip-download", file=sys.stderr)
        sys.exit(1)

    extract_dir = out / "extracted"
    if not args.skip_extract:
        if extract_dir.is_dir() and any(extract_dir.iterdir()):
            print(f"[skip extract] non-empty {extract_dir}")
        else:
            extract_dir.mkdir(parents=True, exist_ok=True)
            print(f"[extract] {zpath} -> {extract_dir}")
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(extract_dir)

    if extract_dir.is_dir():
        try:
            manifest["extracted_subdirs"] = sorted(
                d.name for d in extract_dir.iterdir() if d.is_dir()
            )
        except OSError:
            manifest["extracted_subdirs"] = []

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote", out / "manifest.json")
    if readme.is_file():
        print("Readme:", readme)


if __name__ == "__main__":
    main()
