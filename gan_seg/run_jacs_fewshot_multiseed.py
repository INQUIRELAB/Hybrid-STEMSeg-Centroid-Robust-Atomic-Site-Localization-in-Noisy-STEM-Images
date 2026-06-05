#!/usr/bin/env python3
"""
Run JACS few-shot fine-tuning over a grid: models × n_shot values × seeds.

Defaults match the paper-style grid:
  seeds: 42, 101, 202  (use --seeds 101,202,303 for the alternate triple)
  n_shots: 1, 3, 5, 10

Example:
  python -m gan_seg.run_jacs_fewshot_multiseed
  python -m gan_seg.run_jacs_fewshot_multiseed --seeds 101,202,303 --n-shots 1,3,5,10

Writes:
  reports/jacs_fewshot/jacs_fewshot_grid_runs.csv          (one row per run)
  reports/jacs_fewshot/jacs_fewshot_grid_aggregate.csv     (mean ± std over seeds; per model × n_shot)
  reports/jacs_fewshot/jacs_fewshot_grid_aggregate.json

After the grid, paired tests (Hybrid vs others on held-out images, per seed):
  python -m gan_seg.jacs_fewshot_pairwise_stats
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from gan_seg.jacs_data import DEFAULT_JACS_EXTRACTED, discover_pairs
from gan_seg.train_jacs_fewshot import REPORT_DIR, build_arg_parser, run_fewshot_training


def _run_namespace(
    model: str,
    seed: int,
    n_shot: int,
    epochs: int,
    patience: int,
    lr: float,
    quiet: bool,
    extracted: Path | str,
) -> argparse.Namespace:
    argv = [
        "--model",
        model,
        "--n-shot",
        str(n_shot),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--lr",
        str(lr),
        "--extracted",
        str(Path(extracted)),
    ]
    if quiet:
        argv.append("--quiet-multiseed")
    return build_arg_parser().parse_args(argv)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        type=str,
        default="hybrid-nogan,unet,deeplabv3plus,segformer",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="42,101,202",
        help="Comma-separated RNG seeds",
    )
    p.add_argument(
        "--alt-seeds",
        action="store_true",
        help="Use seeds 101,202,303 (overrides --seeds if both are given)",
    )
    p.add_argument(
        "--n-shots",
        type=str,
        default="1,3,5,10",
        help="Comma-separated shot counts K (each uses K train frames, rest val)",
    )
    p.add_argument(
        "--n-shot",
        type=int,
        default=None,
        help="If set (single int), overrides --n-shots to a single K (legacy)",
    )
    p.add_argument("--extracted", type=str, default=str(DEFAULT_JACS_EXTRACTED))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-epoch logs (default: quiet)",
    )
    cli = p.parse_args()
    quiet = not cli.verbose

    seeds_arg = cli.seeds if not cli.alt_seeds else "101,202,303"
    seeds = [int(x.strip()) for x in seeds_arg.split(",") if x.strip()]
    if cli.n_shot is not None:
        n_shots = [cli.n_shot]
    else:
        n_shots = [int(x.strip()) for x in cli.n_shots.split(",") if x.strip()]
    models = [m.strip() for m in cli.models.split(",") if m.strip()]

    pairs = discover_pairs(Path(cli.extracted))
    n_pairs = len(pairs)
    for k in n_shots:
        if k < 1 or k > n_pairs - 1:
            raise SystemExit(
                f"n_shot={k} invalid: need 1 <= K <= {n_pairs - 1} (have {n_pairs} labeled frames)"
            )

    all_runs: list[dict] = []
    for n_shot in n_shots:
        for model in models:
            for seed in seeds:
                run_args = _run_namespace(
                    model,
                    seed,
                    n_shot,
                    cli.epochs,
                    cli.patience,
                    cli.lr,
                    quiet,
                    Path(cli.extracted),
                )
                print(f"=== n_shot={n_shot} {model} seed={seed} ===", flush=True)
                summ = run_fewshot_training(run_args, quiet=quiet)
                all_runs.append(
                    {
                        "model": summ["model"],
                        "n_shot": summ["n_shot"],
                        "seed": summ["seed"],
                        "n_train": summ["n_train"],
                        "n_val": summ["n_val"],
                        "baseline_f1": summ["val_baseline_mean_f1"],
                        "finetuned_f1": summ["val_finetuned_mean_f1"],
                        "baseline_iou": summ["val_baseline_mean_iou"],
                        "finetuned_iou": summ["val_finetuned_mean_iou"],
                        "best_epoch": summ["best_epoch"],
                    }
                )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    runs_path = REPORT_DIR / "jacs_fewshot_grid_runs.csv"
    with runs_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_runs[0].keys()))
        w.writeheader()
        w.writerows(all_runs)

    agg_rows: list[dict] = []
    for n_shot in n_shots:
        for model in models:
            sub = [r for r in all_runs if r["model"] == model and r["n_shot"] == n_shot]
            if not sub:
                continue
            f1s = [r["finetuned_f1"] for r in sub]
            ious = [r["finetuned_iou"] for r in sub]
            agg_rows.append(
                {
                    "model": model,
                    "n_shot": n_shot,
                    "n_seeds": len(sub),
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                    "iou_mean": float(np.mean(ious)),
                    "iou_std": float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0,
                    "seeds_str": seeds_arg,
                }
            )

    agg_path = REPORT_DIR / "jacs_fewshot_grid_aggregate.csv"
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)

    json_path = REPORT_DIR / "jacs_fewshot_grid_aggregate.json"
    json_path.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "n_shots": n_shots,
                "models": models,
                "n_pairs_jacs": n_pairs,
                "aggregate": agg_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {runs_path} ({len(all_runs)} runs)")
    print(f"Wrote {agg_path}")
    for r in sorted(agg_rows, key=lambda x: (x["n_shot"], x["model"])):
        print(
            f"  K={r['n_shot']} {r['model']}: F1 = {r['f1_mean']:.4f} ± {r['f1_std']:.4f} | "
            f"IoU = {r['iou_mean']:.4f} ± {r['iou_std']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
