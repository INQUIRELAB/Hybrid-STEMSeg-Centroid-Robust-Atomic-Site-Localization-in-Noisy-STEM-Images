#!/usr/bin/env python3
"""
Paired statistical tests for JACS few-shot external adaptation.

Uses per-image finetuned F1 / IoU from val_metrics_{model}_n{K}_s{S}.csv (written by
train_jacs_fewshot). For each n_shot K, RNG seed, and comparator model, aligns images
on (category, image_id) and tests whether Hybrid-STEMSeg differs from the comparator on
the same held-out validation frames.

Methods (per seed, per metric):
  - Paired Wilcoxon signed-rank (two-sided) on image-level paired differences.
  - Bootstrap 95% CI (percentile) for the mean paired difference (Hybrid − comparator),
    resampling validation images with replacement.

Also aggregates across seeds for each (K, comparator):
  - Mean of per-seed mean paired differences with bootstrap CI (resample seeds).
  - Fisher's method to combine the per-seed Wilcoxon p-values (exploratory summary).

Example:
  python -m gan_seg.jacs_fewshot_pairwise_stats
  python -m gan_seg.jacs_fewshot_pairwise_stats --highlight-k 1,5 --n-boot 20000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2, wilcoxon

from gan_seg.train_jacs_fewshot import REPORT_DIR

VAL_METRICS_RE = re.compile(
    r"^val_metrics_(?P<model>[\w-]+)_n(?P<n>\d+)_s(?P<seed>\d+)\.csv$"
)


def _parse_val_metrics_name(name: str) -> dict[str, Any] | None:
    m = VAL_METRICS_RE.match(name)
    if not m:
        return None
    return {
        "model": m.group("model"),
        "n_shot": int(m.group("n")),
        "seed": int(m.group("seed")),
    }


def _load_metrics_csv(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["category"], row["image_id"])
            out[key] = {"f1": float(row["f1"]), "iou": float(row["iou"])}
    return out


def _paired_diffs(
    base: dict[tuple[str, str], dict[str, float]],
    other: dict[tuple[str, str], dict[str, float]],
    metric: str,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    keys = sorted(set(base) & set(other))
    d = np.array([base[k][metric] - other[k][metric] for k in keys], dtype=np.float64)
    return d, keys


def _wilcoxon_p(d: np.ndarray) -> float:
    if d.size == 0:
        return float("nan")
    if np.allclose(d, 0.0):
        return 1.0
    # scipy 1.10+: zero_method default handles zeros in diffs
    res = wilcoxon(d, alternative="two-sided", method="auto")
    p = float(res.pvalue)
    return min(1.0, max(p, 0.0))


def _bootstrap_mean_ci(
    x: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (point_mean, ci_low, ci_high) for mean(x) via bootstrap resampling rows."""
    n = x.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(x))
    if n == 1:
        return point, point, point
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(np.mean(x[idx]))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def _fisher_combine(pvals: list[float]) -> float:
    """Combine independent p-values (Fisher). Returns nan if empty."""
    clean = [p for p in pvals if not math.isnan(p) and p > 0.0]
    if not clean:
        return float("nan")
    chi2_stat = -2.0 * sum(math.log(max(p, 1e-300)) for p in clean)
    df = 2 * len(clean)
    return float(1.0 - chi2.cdf(chi2_stat, df))


def discover_runs(report_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for p in report_dir.glob("val_metrics_*.csv"):
        meta = _parse_val_metrics_name(p.name)
        if meta is None:
            continue
        runs.append({"path": p, **meta})
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    ap.add_argument("--baseline", type=str, default="hybrid-nogan")
    ap.add_argument(
        "--comparators",
        type=str,
        default="unet,deeplabv3plus,segformer",
        help="Comma-separated models to test against --baseline",
    )
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42, help="RNG for bootstrap resamples")
    ap.add_argument(
        "--highlight-k",
        type=str,
        default="1,5",
        help="Comma-separated K values to print as a short summary table",
    )
    cli = ap.parse_args()
    report_dir = cli.report_dir
    comparators = [c.strip() for c in cli.comparators.split(",") if c.strip()]
    highlight_ks = {int(x.strip()) for x in cli.highlight_k.split(",") if x.strip()}
    rng = np.random.default_rng(cli.seed)

    by_triple: dict[tuple[int, int, str], Path] = {}
    for r in discover_runs(report_dir):
        by_triple[(r["n_shot"], r["seed"], r["model"])] = r["path"]

    n_shots = sorted({k[0] for k in by_triple})
    seeds = sorted({k[1] for k in by_triple})
    models_needed = {cli.baseline, *comparators}

    per_seed_rows: list[dict[str, Any]] = []
    for n_shot in n_shots:
        for seed in seeds:
            base_path = by_triple.get((n_shot, seed, cli.baseline))
            if base_path is None:
                continue
            base_m = _load_metrics_csv(base_path)
            for comp in comparators:
                opath = by_triple.get((n_shot, seed, comp))
                if opath is None:
                    continue
                other_m = _load_metrics_csv(opath)
                row: dict[str, Any] = {
                    "n_shot": n_shot,
                    "seed": seed,
                    "baseline": cli.baseline,
                    "comparator": comp,
                    "n_images_paired": 0,
                }
                for metric in ("f1", "iou"):
                    d, keys = _paired_diffs(base_m, other_m, metric)
                    row["n_images_paired"] = len(keys)
                    row[f"{metric}_mean_diff_hybrid_minus_comp"] = float(np.mean(d)) if d.size else float("nan")
                    row[f"{metric}_wilcoxon_p_two_sided"] = _wilcoxon_p(d)
                    pt, lo, hi = _bootstrap_mean_ci(d, cli.n_boot, rng)
                    row[f"{metric}_mean_diff_point"] = pt
                    row[f"{metric}_boot95_ci_low"] = lo
                    row[f"{metric}_boot95_ci_high"] = hi
                per_seed_rows.append(row)

    aggregate: list[dict[str, Any]] = []
    for n_shot in n_shots:
        for comp in comparators:
            sub = [r for r in per_seed_rows if r["n_shot"] == n_shot and r["comparator"] == comp]
            if not sub:
                continue
            item: dict[str, Any] = {
                "n_shot": n_shot,
                "baseline": cli.baseline,
                "comparator": comp,
                "n_seeds": len(sub),
                "seeds": sorted({r["seed"] for r in sub}),
            }
            for metric in ("f1", "iou"):
                ps = [r[f"{metric}_wilcoxon_p_two_sided"] for r in sub]
                mds = np.array(
                    [r[f"{metric}_mean_diff_hybrid_minus_comp"] for r in sub],
                    dtype=np.float64,
                )
                item[f"{metric}_fisher_combined_p"] = _fisher_combine(ps)
                if mds.size == 0:
                    item[f"{metric}_mean_of_seed_mean_diffs"] = float("nan")
                    item[f"{metric}_across_seeds_boot95_low"] = float("nan")
                    item[f"{metric}_across_seeds_boot95_high"] = float("nan")
                else:
                    item[f"{metric}_mean_of_seed_mean_diffs"] = float(np.mean(mds))
                    if mds.size == 1:
                        item[f"{metric}_across_seeds_boot95_low"] = float(mds[0])
                        item[f"{metric}_across_seeds_boot95_high"] = float(mds[0])
                    else:
                        boots = np.empty(cli.n_boot, dtype=np.float64)
                        n_s = mds.size
                        for b in range(cli.n_boot):
                            idx = rng.integers(0, n_s, size=n_s)
                            boots[b] = float(np.mean(mds[idx]))
                        lo, hi = np.quantile(boots, [0.025, 0.975])
                        item[f"{metric}_across_seeds_boot95_low"] = float(lo)
                        item[f"{metric}_across_seeds_boot95_high"] = float(hi)
                item[f"{metric}_wilcoxon_p_per_seed"] = {
                    str(r["seed"]): r[f"{metric}_wilcoxon_p_two_sided"] for r in sorted(sub, key=lambda x: x["seed"])
                }
            aggregate.append(item)

    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "jacs_fewshot_pairwise_stats_per_seed.csv"
    if per_seed_rows:
        fields = list(per_seed_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(sorted(per_seed_rows, key=lambda r: (r["n_shot"], r["comparator"], r["seed"])))

    json_path = report_dir / "jacs_fewshot_pairwise_stats.json"
    payload = {
        "baseline": cli.baseline,
        "comparators": comparators,
        "n_bootstrap": cli.n_boot,
        "bootstrap_seed": cli.seed,
        "per_seed_csv": str(csv_path.name),
        "aggregate": aggregate,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    agg_csv = report_dir / "jacs_fewshot_pairwise_stats_aggregate.csv"
    if aggregate:
        flat_fields = [
            "n_shot",
            "baseline",
            "comparator",
            "n_seeds",
            "f1_mean_of_seed_mean_diffs",
            "f1_across_seeds_boot95_low",
            "f1_across_seeds_boot95_high",
            "f1_fisher_combined_p",
            "iou_mean_of_seed_mean_diffs",
            "iou_across_seeds_boot95_low",
            "iou_across_seeds_boot95_high",
            "iou_fisher_combined_p",
        ]
        with agg_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=flat_fields, extrasaction="ignore")
            w.writeheader()
            for a in sorted(aggregate, key=lambda x: (x["n_shot"], x["comparator"])):
                w.writerow(a)

    print(f"Wrote {csv_path} ({len(per_seed_rows)} rows)")
    print(f"Wrote {json_path}")
    if aggregate:
        print(f"Wrote {agg_csv}")

    # Compact table for paper / K=1 and K=5
    for k in sorted(highlight_ks):
        block = [a for a in aggregate if a["n_shot"] == k]
        if not block:
            continue
        print(f"\n=== K={k}  (Hybrid-STEMSeg vs comparator; + = Hybrid better) ===")
        for a in sorted(block, key=lambda x: x["comparator"]):
            c = a["comparator"]
            print(
                f"  vs {c}: F1  Δ_mean={a['f1_mean_of_seed_mean_diffs']:+.4f} "
                f"95%CI_across_seeds=[{a['f1_across_seeds_boot95_low']:.4f}, {a['f1_across_seeds_boot95_high']:.4f}] "
                f"Fisher p={a['f1_fisher_combined_p']:.4g} | "
                f"IoU Δ_mean={a['iou_mean_of_seed_mean_diffs']:+.4f} "
                f"95%CI=[{a['iou_across_seeds_boot95_low']:.4f}, {a['iou_across_seeds_boot95_high']:.4f}] "
                f"Fisher p={a['iou_fisher_combined_p']:.4g}"
            )
            ps_f1 = a["f1_wilcoxon_p_per_seed"]
            ps_iou = a["iou_wilcoxon_p_per_seed"]
            print(f"         Wilcoxon p per seed (F1): {ps_f1}")
            print(f"         Wilcoxon p per seed (IoU): {ps_iou}")


if __name__ == "__main__":
    main()
