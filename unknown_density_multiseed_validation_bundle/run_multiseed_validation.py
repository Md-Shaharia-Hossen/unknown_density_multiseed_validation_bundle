#!/usr/bin/env python3
"""Evaluate online density estimation and detection performance across random seeds.

Usage:
    python run_multiseed_validation.py
    python run_multiseed_validation.py --seeds 30 --start-seed 100
    python run_multiseed_validation.py --model /path/to/model.py --seeds 20

The hidden true density is used only for evaluation/scoring. The controllers use
observation-derived density estimates, as implemented in the supplied model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def load_model(path: Path):
    spec = importlib.util.spec_from_file_location("coverage_model_for_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load simulation model: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_map(values):
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 1e-12:
        return np.full(arr.shape, 1.0 / arr.size)
    return arr / total


def centroid(probability_map):
    p = normalize_map(probability_map)
    yy, xx = np.indices(p.shape, dtype=float)
    return np.array([np.sum(xx * p), np.sum(yy * p)])


def map_metrics(estimate, truth):
    """Metrics compare normalized probability mass per grid cell."""
    p = normalize_map(estimate)
    q = normalize_map(truth)
    delta = p - q
    pf, qf = p.ravel(), q.ravel()
    corr = np.nan
    if np.std(pf) > 1e-15 and np.std(qf) > 1e-15:
        corr = float(np.corrcoef(pf, qf)[0, 1])
    return {
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "l1_distance": float(np.sum(np.abs(delta))),
        # scipy returns the square root of JS divergence; square it for divergence.
        "js_divergence": float(jensenshannon(pf, qf, base=np.e) ** 2),
        "spatial_correlation": corr,
        "centroid_error_cells": float(np.linalg.norm(centroid(p) - centroid(q))),
        "estimate_entropy_nats": float(-np.sum(pf * np.log(pf + 1e-15))),
        "true_entropy_nats": float(-np.sum(qf * np.log(qf + 1e-15))),
    }


def finite_mean(values):
    a = np.asarray(values, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan


def finite_std(values):
    a = np.asarray(values, dtype=float)
    n = int(np.isfinite(a).sum())
    if n >= 2:
        return float(np.nanstd(a, ddof=1))
    return 0.0 if n == 1 else np.nan


def grouped_summary(df, group_columns, metrics):
    rows = []
    for key, group in df.groupby(group_columns, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key))
        for metric in metrics:
            row[metric + "_mean"] = finite_mean(group[metric])
            row[metric + "_std"] = finite_std(group[metric])
            row[metric + "_n"] = int(np.isfinite(group[metric].to_numpy(dtype=float)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None,
                        help="Simulation source file; defaults to the bundled model.")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of independent seeds (default: 10; minimum: 2).")
    parser.add_argument("--start-seed", type=int, default=10,
                        help="First seed; consecutive seeds are used.")
    parser.add_argument("--output-dir", type=Path, default=Path("multiseed_validation_results"))
    parser.add_argument("--keep-seed-outputs", action="store_true",
                        help="Retain each seed's native model CSVs and plots.")
    args = parser.parse_args()

    if args.seeds < 2:
        parser.error("--seeds must be at least 2.")
    script_dir = Path(__file__).resolve().parent
    model_path = args.model.resolve() if args.model else (
        script_dir / "adaptive_multirobot_networked_coverage_revised.py"
    )
    if not model_path.exists():
        parser.error(f"Model source not found: {model_path}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_root = out_dir / "per_seed"
    if seed_root.exists():
        shutil.rmtree(seed_root)
    seed_root.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    density_rows, score_rows, seed_rows = [], [], []
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    start = time.time()

    print(f"Model: {model_path}")
    print(f"Seeds: {seeds[0]} through {seeds[-1]} ({len(seeds)} runs)")
    for run_i, seed in enumerate(seeds, 1):
        print(f"[{run_i}/{len(seeds)}] seed={seed}", flush=True)
        model.RANDOM_SEED = int(seed)
        seed_dir = seed_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model.OUTPUT_DIR = seed_dir

        result = model.run_simulation()
        estimates = np.asarray(result["density_estimate_paths"], dtype=float)
        if len(estimates) < int(model.TIME_STEPS) + 1:
            raise RuntimeError(
                f"Seed {seed}: expected >= {model.TIME_STEPS + 1} density estimates; "
                f"got {len(estimates)}."
            )

        # estimate_paths[0] is the initial uniform prior. estimate_paths[t+1]
        # includes observations collected at t, so compare it with goal_density(t).
        for t in range(int(model.TIME_STEPS)):
            row = {"seed": seed, "time": t}
            row.update(map_metrics(estimates[t + 1], model.goal_density(t)))
            density_rows.append(row)

        for method, values in result["scores"].items():
            for t, score in enumerate(values):
                score_rows.append({
                    "seed": seed,
                    "time": int(t),
                    "method": str(method),
                    "detection_score": float(score),
                })

        seed_row = {"seed": seed}
        for method, values in result["scores"].items():
            seed_row["mean_score__" + method] = finite_mean(values)
        seed_row["mean_objective_robot_step"] = finite_mean(
            [r["objective_mean_robot_step"] for r in result["movement_log"]]
        )
        seed_row["total_density_gradient_travel"] = float(
            sum(r["density_gradient_step_travel"] for r in result["movement_log"])
        )
        seed_row["total_objective_gradient_travel"] = float(
            sum(r["objective_gradient_step_travel"] for r in result["movement_log"])
        )
        seed_rows.append(seed_row)

        if not args.keep_seed_outputs:
            shutil.rmtree(seed_dir, ignore_errors=True)

    density_df = pd.DataFrame(density_rows)
    score_df = pd.DataFrame(score_rows)
    seed_df = pd.DataFrame(seed_rows)

    density_metrics = [
        "mae", "rmse", "l1_distance", "js_divergence", "spatial_correlation",
        "centroid_error_cells", "estimate_entropy_nats", "true_entropy_nats",
    ]
    density_df.to_csv(out_dir / "density_metrics_by_seed_time.csv", index=False)
    score_df.to_csv(out_dir / "detection_scores_by_seed_time.csv", index=False)
    seed_df.to_csv(out_dir / "per_seed_mean_scores.csv", index=False)

    density_time_summary = grouped_summary(density_df, ["time"], density_metrics)
    density_time_summary.to_csv(out_dir / "density_metrics_over_time_mean_std.csv", index=False)
    density_seed_summary = grouped_summary(density_df, ["seed"], density_metrics)
    density_seed_summary.to_csv(out_dir / "density_metrics_summary_by_seed.csv", index=False)

    score_time_summary = grouped_summary(score_df, ["time", "method"], ["detection_score"])
    score_time_summary.to_csv(out_dir / "detection_scores_over_time_mean_std.csv", index=False)

    score_long = seed_df.melt(
        id_vars=["seed"],
        value_vars=[c for c in seed_df.columns if c.startswith("mean_score__")],
        var_name="method", value_name="time_averaged_detection_score",
    )
    score_long["method"] = score_long["method"].str.replace("mean_score__", "", regex=False)
    score_across_seeds = grouped_summary(
        score_long, ["method"], ["time_averaged_detection_score"]
    )
    score_across_seeds.to_csv(out_dir / "detection_scores_across_seeds_mean_std.csv", index=False)

    # Density-estimation error curves (mean +/- one sample SD across seeds).
    plot_specs = [
        ("rmse", "RMSE (normalized probability mass per cell)", "density_rmse_over_time.png",
         "Online target-density estimation RMSE"),
        ("js_divergence", "Jensen-Shannon divergence (nats)", "density_js_divergence_over_time.png",
         "Online target-density estimation divergence"),
        ("centroid_error_cells", "Centroid error (grid cells)", "density_centroid_error_over_time.png",
         "Estimated versus true target-density centroid"),
    ]
    for metric, ylabel, filename, title in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = density_time_summary["time"].to_numpy()
        mean = density_time_summary[metric + "_mean"].to_numpy()
        sd = density_time_summary[metric + "_std"].to_numpy()
        ax.plot(x, mean, linewidth=2, label="Mean across seeds")
        ax.fill_between(x, np.maximum(0, mean - sd), mean + sd, alpha=0.2,
                        label="Mean ± 1 SD")
        ax.set_xlabel("Time step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    # Detection score curves by policy (mean +/- one sample SD across seeds).
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method in score_df["method"].drop_duplicates():
        part = score_time_summary[score_time_summary["method"] == method].sort_values("time")
        x = part["time"].to_numpy()
        mean = part["detection_score_mean"].to_numpy()
        sd = part["detection_score_std"].to_numpy()
        ax.plot(x, mean, linewidth=2, label=method)
        ax.fill_between(x, mean - sd, mean + sd, alpha=0.12)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Detection score")
    ax.set_title(f"Detection performance across {args.seeds} random seeds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "detection_scores_over_time.png", dpi=180)
    plt.close(fig)

    # Overall per-seed score comparison, mean +/- sample SD.
    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = score_across_seeds["method"].to_list()
    means = score_across_seeds["time_averaged_detection_score_mean"].to_numpy()
    sds = score_across_seeds["time_averaged_detection_score_std"].to_numpy()
    ax.bar(labels, means, yerr=sds, capsize=4)
    ax.set_ylabel("Time-averaged detection score")
    ax.set_title(f"Detection score by policy across {args.seeds} seeds (mean ± SD)")
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "detection_scores_across_seeds.png", dpi=180)
    plt.close(fig)

    metadata = {
        "model_source": str(model_path),
        "seeds": seeds,
        "num_seeds": len(seeds),
        "grid_size": int(model.GRID_SIZE),
        "num_robots": int(model.NUM_ROBOTS),
        "time_steps": int(model.TIME_STEPS),
        "target_sigma": float(model.TARGET_SIGMA),
        "estimator_bandwidth": float(model.ESTIMATOR_BANDWIDTH),
        "estimator_memory": float(model.ESTIMATOR_MEMORY),
        "estimator_prior_weight": float(model.ESTIMATOR_PRIOR_WEIGHT),
        "elapsed_seconds": round(time.time() - start, 3),
        "metric_note": (
            "Density maps are normalized to sum to one. MAE/RMSE are per-cell probability "
            "mass errors. JS divergence uses natural logarithms. Estimate at t+1 includes "
            "observations from t and is compared with the hidden true map at t."
        ),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== Time-averaged detection score across seeds (mean ± SD) ===")
    print(score_across_seeds.to_string(index=False))
    print("\n=== Density metrics averaged over all seed/time rows ===")
    for metric in ["mae", "rmse", "l1_distance", "js_divergence",
                   "spatial_correlation", "centroid_error_cells"]:
        print(f"{metric:24s} mean={finite_mean(density_df[metric]):.6g}, "
              f"SD={finite_std(density_df[metric]):.6g}")
    print(f"\nCompleted in {time.time() - start:.1f} seconds.")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
