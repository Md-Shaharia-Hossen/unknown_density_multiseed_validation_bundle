
"""
Ablation Study: Adaptive Multi-Robot Probabilistic Area Coverage

Files required in the same directory:
    ablation_study.py
    adaptive_multirobot_networked_coverage_revised.py

Run:
    python ablation_study.py

Outputs:
    ablation_results/ablation_results_by_seed.csv
    ablation_results/ablation_summary.csv
    ablation_results/ablation_paired_differences.csv
    ablation_results/ablation_detection_timeseries.csv
    ablation_results/ablation_travel_timeseries.csv
    ablation_results/ablation_figure.png
    ablation_results/run_metadata.json

Dependencies:
    numpy, pandas, scipy, matplotlib
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

import adaptive_multirobot_networked_coverage_revised as sim


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

N_SEEDS = 30
SEEDS = list(range(100, 100 + N_SEEDS))

OUTPUT_DIR = Path("ablation_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_ROBOTS = sim.NUM_ROBOTS
N_STEPS = sim.TIME_STEPS
MAX_STEP = sim.MAX_STEP_OBJECTIVE
STEP_GAIN = sim.OBJECTIVE_STEP_GAIN


@dataclass(frozen=True)
class AblationConfig:
    name: str
    controller: str = "detection_objective"
    use_backtracking: bool = True
    estimator_memory: float = 0.85
    adaptive_step_scaling: bool = True


CONFIGS = [
    AblationConfig(
        name="A0_full_method",
        controller="detection_objective",
        use_backtracking=True,
        estimator_memory=0.85,
        adaptive_step_scaling=True,
    ),
    AblationConfig(
        name="A1_density_gradient",
        controller="density_gradient",
        use_backtracking=False,
        estimator_memory=0.85,
        adaptive_step_scaling=False,
    ),
    AblationConfig(
        name="A2_no_backtracking",
        controller="detection_objective",
        use_backtracking=False,
        estimator_memory=0.85,
        adaptive_step_scaling=True,
    ),
    AblationConfig(
        name="A3_no_temporal_smoothing",
        controller="detection_objective",
        use_backtracking=True,
        estimator_memory=0.0,
        adaptive_step_scaling=True,
    ),
    AblationConfig(
        name="A4_fixed_step_scaling",
        controller="detection_objective",
        use_backtracking=True,
        estimator_memory=0.85,
        adaptive_step_scaling=False,
    ),
]


# ============================================================
# 2. DENSITY ESTIMATOR WITH CONFIGURABLE MEMORY
# ============================================================

def update_estimate(previous_estimate, observations, memory):
    """
    Update the estimated density using observed target coordinates.

    memory=0.85: temporal smoothing.
    memory=0.0: no temporal smoothing.

    The simulator's uniform prior is retained to match the original model.
    """
    uniform = np.full(
        (sim.GRID_SIZE, sim.GRID_SIZE),
        1.0 / (sim.GRID_SIZE * sim.GRID_SIZE),
    )

    current = sim.estimate_density_from_observations(observations)

    if current.sum() <= sim.EPSILON:
        updated = (
            memory * previous_estimate
            + (1.0 - memory) * uniform
        )
    else:
        updated = (
            memory * previous_estimate
            + (1.0 - memory) * current
        )

    updated = (
        (1.0 - sim.ESTIMATOR_PRIOR_WEIGHT) * updated
        + sim.ESTIMATOR_PRIOR_WEIGHT * uniform
    )

    total = float(updated.sum())

    if total <= sim.EPSILON:
        return uniform

    return updated / total


# ============================================================
# 3. CONFIGURABLE OBJECTIVE-GRADIENT CONTROLLER
# ============================================================

def objective_controller(
    robot_positions,
    density,
    max_step,
    use_backtracking=True,
    adaptive_step_scaling=True,
):
    """
    Objective-gradient controller.

    Ablation switches:
      use_backtracking:
          True  -> accept only non-decreasing local objective moves.
          False -> accept the proposed gradient step directly.

      adaptive_step_scaling:
          True  -> step length = min(max_step, STEP_GAIN * gradient_norm).
          False -> use max_step whenever the gradient is nonzero.
    """
    updated = np.asarray(robot_positions, dtype=float).copy()
    accepted_steps = np.zeros(len(updated), dtype=float)

    for i in range(len(updated)):
        adjacency = sim.communication_adjacency(
            updated, sim.COMMUNICATION_RANGE
        )

        component = next(
            c for c in sim.connected_components(adjacency)
            if i in c
        )

        gradient = sim.objective_gradient_for_robot(
            i, updated, density, component
        )

        norm = float(np.linalg.norm(gradient))

        if norm <= sim.GRADIENT_DEADBAND:
            continue

        direction = gradient / norm

        if adaptive_step_scaling:
            proposed_distance = min(max_step, STEP_GAIN * norm)
        else:
            proposed_distance = max_step

        if proposed_distance <= 1e-12:
            continue

        # No-backtracking ablation.
        if not use_backtracking:
            candidate = np.clip(
                updated[i] + proposed_distance * direction,
                0.0,
                sim.GRID_SIZE - 1.0,
            )

            accepted_steps[i] = float(
                np.linalg.norm(candidate - updated[i])
            )
            updated[i] = candidate
            continue

        # Full method: local objective acceptance with backtracking.
        baseline = sim.component_detection_score(
            updated, density, component
        )

        distance = proposed_distance

        for _ in range(12):
            candidate = updated.copy()

            candidate[i] = np.clip(
                updated[i] + distance * direction,
                0.0,
                sim.GRID_SIZE - 1.0,
            )

            actual_distance = float(
                np.linalg.norm(candidate[i] - updated[i])
            )

            if actual_distance <= 1e-12:
                distance *= 0.5
                continue

            candidate_score = sim.component_detection_score(
                candidate, density, component
            )

            if candidate_score >= baseline - 1e-12:
                updated = candidate
                accepted_steps[i] = actual_distance
                break

            distance *= 0.5

    return updated, accepted_steps


# ============================================================
# 4. RUN ONE SIMULATION
# ============================================================

def run_one_experiment(seed, config):
    """
    Run one ablation configuration for one seed.

    Each policy maintains its own robot positions, observations,
    density estimator, and random generator.

    True density is used only for observation generation and evaluation.
    """
    rng = np.random.default_rng(seed)

    uniform = np.full(
        (sim.GRID_SIZE, sim.GRID_SIZE),
        1.0 / (sim.GRID_SIZE * sim.GRID_SIZE),
    )

    # Fair common starting configuration for adaptive controllers.
    positions = sim.greedy_initial_positions(uniform, N_ROBOTS)

    estimate = uniform.copy()

    detection_scores = []
    estimated_scores = []
    cumulative_travel_series = []

    cumulative_travel = 0.0

    started = time.perf_counter()

    for step in range(N_STEPS):
        # Hidden true field.
        true_density = sim.goal_density(step)

        # Important: observations are generated using THIS variant's
        # robot positions, not positions from another controller.
        observations = sim.observe_targets(
            true_density,
            positions,
            rng,
        )

        # The controller receives only the estimated density.
        estimate = update_estimate(
            estimate,
            observations,
            config.estimator_memory,
        )

        # Evaluate current positions using the hidden true density.
        detection_scores.append(
            sim.detection_score(positions, true_density)
        )

        estimated_scores.append(
            sim.detection_score(positions, estimate)
        )

        # Apply the selected policy.
        if config.controller == "density_gradient":
            positions, step_distances = (
                sim.density_gradient_controller(
                    positions,
                    estimate,
                    sim.MAX_STEP_DENSITY,
                )
            )

        elif config.controller == "detection_objective":
            positions, step_distances = objective_controller(
                positions,
                estimate,
                MAX_STEP,
                use_backtracking=config.use_backtracking,
                adaptive_step_scaling=config.adaptive_step_scaling,
            )

        else:
            raise ValueError(
                f"Unknown controller: {config.controller}"
            )

        cumulative_travel += float(np.sum(step_distances))
        cumulative_travel_series.append(cumulative_travel)

    elapsed = time.perf_counter() - started

    return {
        "scores": np.asarray(detection_scores, dtype=float),
        "estimated_scores": np.asarray(estimated_scores, dtype=float),
        "cumulative_travel": float(cumulative_travel),
        "travel_series": np.asarray(
            cumulative_travel_series, dtype=float
        ),
        "runtime_seconds": float(elapsed),
    }


# ============================================================
# 5. STATISTICAL SUMMARIES
# ============================================================

def mean_ci95(values):
    """Student-t 95% confidence interval across independent seeds."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(x))

    if len(x) < 2:
        return mean, np.nan, np.nan

    sem = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    half = float(
        student_t.ppf(0.975, len(x) - 1) * sem
    )

    return mean, mean - half, mean + half


# ============================================================
# 6. RUN THE FULL ABLATION STUDY
# ============================================================

def run_study():
    result_rows = []
    time_rows = []
    travel_rows = []

    started = time.perf_counter()

    metadata = {
        "seeds": SEEDS,
        "number_of_seeds": N_SEEDS,
        "number_of_robots": N_ROBOTS,
        "number_of_steps": N_STEPS,
        "grid_size": sim.GRID_SIZE,
        "sensor_sigma": sim.SIGMA_SENSOR,
        "target_sigma": sim.TARGET_SIGMA,
        "targets_per_step_expected": sim.TARGETS_PER_STEP,
        "estimator_bandwidth": sim.ESTIMATOR_BANDWIDTH,
        "estimator_prior_weight": sim.ESTIMATOR_PRIOR_WEIGHT,
        "objective_step_gain": STEP_GAIN,
        "configurations": [asdict(c) for c in CONFIGS],
        "notes": [
            "The hidden true density is used only for observations and evaluation.",
            "Each variant has its own observations and density estimator.",
            "Results are aggregated across independent seeds, not time steps.",
            "The baseline A1 is a controller comparison, not a single-component ablation.",
            "Same seed IDs do not guarantee identical observation samples across policies.",
        ],
    }

    for config in CONFIGS:
        print(f"\nVariant: {config.name}", flush=True)

        for seed in SEEDS:
            print(f"  seed={seed}", flush=True)

            result = run_one_experiment(seed, config)
            scores = result["scores"]

            result_rows.append({
                "variant": config.name,
                "seed": seed,
                "controller": config.controller,
                "use_backtracking": config.use_backtracking,
                "estimator_memory": config.estimator_memory,
                "adaptive_step_scaling": config.adaptive_step_scaling,
                "mean_detection": float(np.mean(scores)),
                "std_detection_over_time": float(
                    np.std(scores, ddof=1)
                ),
                "final_20_step_detection": float(
                    np.mean(scores[-20:])
                ),
                "min_detection": float(np.min(scores)),
                "max_detection": float(np.max(scores)),
                "cumulative_travel": result["cumulative_travel"],
                "runtime_seconds": result["runtime_seconds"],
            })

            for step, score in enumerate(scores):
                time_rows.append({
                    "variant": config.name,
                    "seed": seed,
                    "time_step": step,
                    "detection_score": float(score),
                    "estimated_density_score": float(
                        result["estimated_scores"][step]
                    ),
                })

            for step, travel in enumerate(result["travel_series"]):
                travel_rows.append({
                    "variant": config.name,
                    "seed": seed,
                    "time_step": step,
                    "cumulative_travel": float(travel),
                })

    df = pd.DataFrame(result_rows)
    timeseries = pd.DataFrame(time_rows)
    travel_df = pd.DataFrame(travel_rows)

    df.to_csv(
        OUTPUT_DIR / "ablation_results_by_seed.csv",
        index=False,
    )

    timeseries.to_csv(
        OUTPUT_DIR / "ablation_detection_timeseries.csv",
        index=False,
    )

    travel_df.to_csv(
        OUTPUT_DIR / "ablation_travel_timeseries.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary with confidence intervals across seeds
    # --------------------------------------------------------
    summary_rows = []

    metrics = [
        "mean_detection",
        "final_20_step_detection",
        "cumulative_travel",
        "runtime_seconds",
    ]

    for config in CONFIGS:
        group = df[df["variant"] == config.name]

        row = {
            "variant": config.name,
            "n_seeds": int(group["seed"].nunique()),
        }

        for metric in metrics:
            mean, low, high = mean_ci95(group[metric])

            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
            row[f"{metric}_std_across_seeds"] = float(
                group[metric].std(ddof=1)
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT_DIR / "ablation_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Paired differences by seed relative to A0
    # Positive detection difference means A0 performed better.
    # Positive travel difference means A0 travelled farther.
    # --------------------------------------------------------
    paired_rows = []

    full = (
        df[df["variant"] == "A0_full_method"]
        .set_index("seed")
    )

    for config in CONFIGS:
        if config.name == "A0_full_method":
            continue

        other = (
            df[df["variant"] == config.name]
            .set_index("seed")
        )

        for metric in [
            "mean_detection",
            "final_20_step_detection",
            "cumulative_travel",
        ]:
            differences = (
                full[metric] - other[metric]
            ).dropna()

            mean, low, high = mean_ci95(differences)

            paired_rows.append({
                "comparison": (
                    f"A0_full_method_minus_{config.name}"
                ),
                "metric": metric,
                "n_paired_seeds": len(differences),
                "mean_difference": mean,
                "ci95_low": low,
                "ci95_high": high,
                "difference_std": float(
                    differences.std(ddof=1)
                ),
            })

    pd.DataFrame(paired_rows).to_csv(
        OUTPUT_DIR / "ablation_paired_differences.csv",
        index=False,
    )

    # ========================================================
    # 7. SIX-PANEL FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 9)
    )

    short_labels = [
        "A0 Full",
        "A1 Density gradient",
        "A2 No backtracking",
        "A3 No smoothing",
        "A4 Fixed step",
    ]

    # Panel 1: mean detection over time, with +/- 1 SD across seeds.
    ax = axes[0, 0]

    for config, label in zip(CONFIGS, short_labels):
        sub = timeseries[
            timeseries["variant"] == config.name
        ]

        agg = (
            sub.groupby("time_step")["detection_score"]
            .agg(["mean", "std"])
            .reset_index()
        )

        x = agg["time_step"].to_numpy()
        y = agg["mean"].to_numpy()
        sd = agg["std"].fillna(0).to_numpy()

        ax.plot(x, y, label=label, linewidth=1.5)
        ax.fill_between(
            x,
            np.maximum(0, y - sd),
            np.minimum(1, y + sd),
            alpha=0.12,
        )

    ax.set_title("Detection score over time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Expected detection probability")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)

    # Panel 2: mean cumulative travel over time.
    ax = axes[0, 1]

    for config, label in zip(CONFIGS, short_labels):
        sub = travel_df[
            travel_df["variant"] == config.name
        ]

        agg = (
            sub.groupby("time_step")["cumulative_travel"]
            .agg(["mean", "std"])
            .reset_index()
        )

        x = agg["time_step"].to_numpy()
        y = agg["mean"].to_numpy()
        sd = agg["std"].fillna(0).to_numpy()

        ax.plot(x, y, label=label, linewidth=1.5)
        ax.fill_between(
            x, np.maximum(0, y - sd), y + sd, alpha=0.12
        )

    ax.set_title("Cumulative robot travel")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Distance")
    ax.legend(fontsize=7)

    # Panels 3-6: seed-level metrics with 95% confidence intervals.
    panel_metrics = [
        ("mean_detection", "Mean detection", 0),
        (
            "final_20_step_detection",
            "Final 20-step detection",
            1,
        ),
        ("cumulative_travel", "Total travel", 2),
        ("runtime_seconds", "Runtime per seed (s)", 3),
    ]

    positions = np.arange(len(CONFIGS))

    for metric, title, panel_index in panel_metrics:
        if panel_index == 0:
            ax = axes[0, 2]
        else:
            ax = axes[1, panel_index - 1]

        means, lows, highs = [], [], []

        for config in CONFIGS:
            vals = df.loc[
                df["variant"] == config.name, metric
            ].to_numpy()

            mean, low, high = mean_ci95(vals)

            means.append(mean)
            lows.append(
                max(0.0, mean - low)
                if np.isfinite(low) else 0.0
            )
            highs.append(
                max(0.0, high - mean)
                if np.isfinite(high) else 0.0
            )

        ax.bar(positions, means)
        ax.errorbar(
            positions,
            means,
            yerr=[lows, highs],
            fmt="none",
            capsize=4,
            color="black",
        )

        ax.set_title(f"{title} (95% CI)")
        ax.set_xticks(
            positions,
            ["A0", "A1", "A2", "A3", "A4"],
        )
        ax.tick_params(axis="x", labelsize=9)

        if metric in [
            "mean_detection",
            "final_20_step_detection",
        ]:
            ax.set_ylim(0, 1)

    fig.suptitle(
        "Ablation Study: Adaptive Multi-Robot Probabilistic Coverage",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(
        OUTPUT_DIR / "ablation_figure.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    with open(
        OUTPUT_DIR / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\nAblation summary:")
    print(
        summary[
            [
                "variant",
                "mean_detection_mean",
                "mean_detection_ci95_low",
                "mean_detection_ci95_high",
                "final_20_step_detection_mean",
                "cumulative_travel_mean",
            ]
        ].to_string(index=False)
    )

    print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")
    print(
        f"Total runtime: "
        f"{(time.perf_counter() - started) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    run_study()
