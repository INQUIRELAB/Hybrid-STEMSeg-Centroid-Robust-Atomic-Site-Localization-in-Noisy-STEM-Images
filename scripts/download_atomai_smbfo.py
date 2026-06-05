"""Download public Sm-BFO STEM data via AtomAI (external source, not local .npy)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from atomai.utils import datasets

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "external_stem_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    smbfo = datasets.stem_smbfo(download=True, filedir=str(OUT_DIR))

    keys = list(smbfo.keys())
    manifest = {
        "source": "atomai.utils.datasets.stem_smbfo",
        "filedir": str(OUT_DIR),
        "n_keys": len(keys),
        "keys_sample": keys[:20],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Keys (first 10):", keys[:10])
    print("Total keys:", len(keys))

    key = keys[0]
    entry = smbfo[key]
    img = np.asarray(entry["main_image"], dtype=np.float32)

    # AtomAI dict may use xy_COM or xy_atms depending on version
    if "xy_COM" in entry:
        coords = np.asarray(entry["xy_COM"], dtype=np.float64)
        coord_key = "xy_COM"
    elif "xy_atms" in entry:
        coords = np.asarray(entry["xy_atms"], dtype=np.float64)
        coord_key = "xy_atms"
    else:
        raise KeyError(f"No coordinates in entry; keys: {list(entry.keys())}")

    # Dataset convention: rows are (y, x) for mask building — scatter uses column order
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) coords, got {coords.shape}")

    plt.imsave(str(OUT_DIR / "smbfo_public_stem.png"), img, cmap="gray")
    np.save(OUT_DIR / "smbfo_public_stem.npy", img)
    np.save(OUT_DIR / "smbfo_public_coords.npy", coords)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray")
    ax.scatter(coords[:, 1], coords[:, 0], s=3, c="r", alpha=0.7)
    ax.set_title(f"{key} ({coord_key}: y,x → scatter x,y)")
    ax.axis("off")
    fig.savefig(OUT_DIR / "smbfo_public_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", OUT_DIR / "smbfo_public_stem.png")
    print("Saved:", OUT_DIR / "smbfo_public_stem.npy", img.shape, img.dtype)
    print("Saved:", OUT_DIR / "smbfo_public_coords.npy", coords.shape, coord_key)
    print("Saved:", OUT_DIR / "smbfo_public_overlay.png")


if __name__ == "__main__":
    main()
