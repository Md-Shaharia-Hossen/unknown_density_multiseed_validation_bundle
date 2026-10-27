"""Adaptive Multi-Robot Probabilistic Area Coverage with a Moving Target Field

Rewritten from the user's previous adaptive_multirobot_networked_coverage.py.

WHAT THIS SCRIPT MODELS
-----------------------
* A 50 x 50 workspace and five mobile robots.
* A Gaussian target-density field whose center moves with time.
* Independent probabilistic sensors combined into a team detection probability.
* Four policies: random-static, greedy-static, density-gradient adaptive,
  and detection-objective-gradient adaptive.
* A peer-to-peer communication graph, network statistics, trajectories,
  detection scores, and cumulative travel distance.

IMPORTANT SCIENTIFIC LIMITATIONS
--------------------------------
1. Controllers use an estimated density map built from noisy target detections;
   the true density is used only to generate observations and evaluate performance.
2. This is a simulated unknown-field experiment, not a real-world deployment.
   Its conclusions depend on the observation model and estimator settings.
3. The communication graph is modeled and logged. Since COMMUNICATION_RANGE=80
   exceeds the grid's maximum diagonal, the default graph is effectively complete.
4. The gradient controller is a heuristic local optimizer; no global-optimality
   guarantee is claimed. Results from one seed are not statistical evidence of
   general superiority; run multiple seeds before making paper claims.

Dependencies: numpy, matplotlib
Run: python adaptive_multirobot_networked_coverage_revised.py
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PART 1 — CONFIGURATION
# =============================================================================
GRID_SIZE = 50
NUM_ROBOTS = 5
TIME_STEPS = 200

SIGMA_SENSOR = 4.0       # Sensor footprint width in the detection model
TARGET_SIGMA = 8.0       # Width of the moving target-density Gaussian

MAX_STEP_DENSITY = 1.0   # Maximum movement per robot per time step
MAX_STEP_OBJECTIVE = 1.0

# Maximum distance between two points in a 50 x 50 grid is about 69.3.
# Therefore 80.0 makes the default communication graph complete.
COMMUNICATION_RANGE = 80.0

RANDOM_SEED = 42
EPSILON = 1e-12
GRADIENT_DEADBAND = 1e-10

# Unknown-field observation/estimation settings.
TARGETS_PER_STEP = 100       # Expected number of targets sampled per time step
ESTIMATOR_BANDWIDTH = 3.0   # Gaussian KDE bandwidth in grid cells
ESTIMATOR_MEMORY = 0.85     # Temporal smoothing; larger values retain more history
ESTIMATOR_PRIOR_WEIGHT = 0.15

# The raw objective gradient can be small because the density is normalized.
# This gain maps gradient magnitude to a movement distance. The result is
# clipped at MAX_STEP_OBJECTIVE and can be smaller than the maximum step.
OBJECTIVE_STEP_GAIN = 25.0

OUTPUT_DIR = Path("coverage_results_revised")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

axis = np.arange(GRID_SIZE, dtype=float)
X, Y = np.meshgrid(axis, axis)


# =============================================================================
# PART 2 — MOVING TARGET-DENSITY FIELD
# =============================================================================
def target_center(t):
    """Return the moving Gaussian center (cx, cy) at time t."""
    cx = 10.0 + 25.0 * np.sin(t / 10.0)
    cy = 25.0 + 10.0 * np.cos(t / 15.0)
    # Keep the center inside the workspace, including at the extremes.
    return float(np.clip(cx, 0, GRID_SIZE - 1)), float(np.clip(cy, 0, GRID_SIZE - 1))


def goal_density(t):
    """Return a normalized Gaussian probability mass map for time t."""
    cx, cy = target_center(t)
    density = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * TARGET_SIGMA**2))
    total = float(density.sum())
    if total <= EPSILON:
        raise RuntimeError("Target-density map has zero total mass.")
    return density / total


# =============================================================================
# PART 3 — SENSOR MODEL AND TEAM DETECTION OBJECTIVE
# =============================================================================
def per_robot_detection_probabilities(robot_positions):
    """Return an array with shape (robots, y, x), containing p_i(q)."""
    positions = np.asarray(robot_positions, dtype=float)
    if positions.size == 0:
        return np.zeros((0, GRID_SIZE, GRID_SIZE), dtype=float)

    d2 = (X[None, :, :] - positions[:, 0, None, None]) ** 2
    d2 += (Y[None, :, :] - positions[:, 1, None, None]) ** 2
    return np.exp(-d2 / (2.0 * SIGMA_SENSOR**2))


def sensor_detection(robot_positions):
    """P(at least one robot detects a target at q) = 1 - product(1-p_i)."""
    probabilities = per_robot_detection_probabilities(robot_positions)
    if probabilities.shape[0] == 0:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    return 1.0 - np.prod(1.0 - probabilities, axis=0)


def detection_score(robot_positions, density):
    """Expected team detection probability under normalized target density."""
    return float(np.sum(sensor_detection(robot_positions) * density))


# =============================================================================
# PART 3B — UNKNOWN-FIELD OBSERVATIONS AND DENSITY ESTIMATION
# =============================================================================
def observe_targets(true_density, robot_positions, rng):
    """Return detected target coordinates without exposing the true map to control.

    The simulator samples target locations from the hidden density. A target is
    observed with probability equal to the team's combined sensor detection
    probability at that location. The controller receives only detected points.
    """
    count = int(rng.poisson(TARGETS_PER_STEP))
    if count <= 0:
        return np.empty((0, 2), dtype=float)

    flat_prob = np.asarray(true_density, dtype=float).ravel()
    flat_prob = flat_prob / flat_prob.sum()
    flat_indices = rng.choice(flat_prob.size, size=count, p=flat_prob)
    ys, xs = np.unravel_index(flat_indices, true_density.shape)
    candidates = np.column_stack((xs.astype(float), ys.astype(float)))

    sensor_prob = per_robot_detection_probabilities(robot_positions)
    # Look up each candidate's combined detection probability on the grid.
    p_detect_map = 1.0 - np.prod(1.0 - sensor_prob, axis=0)
    detected = rng.random(count) < p_detect_map[ys, xs]
    return candidates[detected]


def estimate_density_from_observations(observations):
    """Construct a normalized Gaussian-kernel density map from observed points."""
    estimate = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    if len(observations) == 0:
        return estimate
    for ox, oy in observations:
        estimate += np.exp(-((X - ox) ** 2 + (Y - oy) ** 2) /
                           (2.0 * ESTIMATOR_BANDWIDTH ** 2))
    total = float(estimate.sum())
    if total > EPSILON:
        estimate /= total
    return estimate


def update_density_estimate(previous_estimate, observations):
    """Update the map using temporally smoothed KDE and a weak uniform prior."""
    uniform = np.full((GRID_SIZE, GRID_SIZE), 1.0 / (GRID_SIZE * GRID_SIZE))
    current = estimate_density_from_observations(observations)
    if current.sum() <= EPSILON:
        # No detections: retain decayed history and blend in a weak uniform prior.
        updated = ESTIMATOR_MEMORY * previous_estimate + (1.0 - ESTIMATOR_MEMORY) * uniform
    else:
        updated = (ESTIMATOR_MEMORY * previous_estimate +
                   (1.0 - ESTIMATOR_MEMORY) * current)
    updated = (1.0 - ESTIMATOR_PRIOR_WEIGHT) * updated + ESTIMATOR_PRIOR_WEIGHT * uniform
    total = float(updated.sum())
    return updated / total if total > EPSILON else uniform


# =============================================================================
# PART 4 — STATIC GREEDY INITIALIZATION
# =============================================================================
def greedy_initial_positions(density, number_of_robots):
    """Sequentially place robots at cells with the greatest marginal gain.

    This is a simple greedy initializer, not an exact global optimizer.
    A vectorized candidate calculation replaces the slow candidate-by-candidate
    Python loop from the earlier version while preserving its greedy logic.
    """
    candidates = np.column_stack((X.ravel(), Y.ravel()))
    residual = density.copy()
    selected = []

    for _ in range(number_of_robots):
        d2 = (X[None, :, :] - candidates[:, 0, None, None]) ** 2
        d2 += (Y[None, :, :] - candidates[:, 1, None, None]) ** 2
        p = np.exp(-d2 / (2.0 * SIGMA_SENSOR**2))
        gains = np.sum(p * residual[None, :, :], axis=(1, 2))
        best_idx = int(np.argmax(gains))
        best_point = candidates[best_idx].copy()
        selected.append(best_point)

        d2_best = (X - best_point[0]) ** 2 + (Y - best_point[1]) ** 2
        p_best = np.exp(-d2_best / (2.0 * SIGMA_SENSOR**2))
        residual *= (1.0 - p_best)

    return np.asarray(selected, dtype=float)


# =============================================================================
# PART 5 — COMMUNICATION GRAPH
# =============================================================================
def communication_adjacency(robot_positions, communication_range):
    """Undirected adjacency matrix; an edge exists within communication range."""
    positions = np.asarray(robot_positions, dtype=float)
    n = len(positions)
    adjacency = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) <= communication_range:
                adjacency[i, j] = adjacency[j, i] = 1
    return adjacency


def connected_components(adjacency):
    """Return connected components as lists of robot indices."""
    n = adjacency.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in np.flatnonzero(adjacency[node]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        components.append(sorted(component))
    return components


def network_statistics(adjacency):
    """Summarize graph connectivity, links, average degree, and components."""
    components = connected_components(adjacency)
    degrees = adjacency.sum(axis=1)
    return {
        "connected": int(len(components) <= 1),
        "edges": int(np.triu(adjacency, k=1).sum()),
        "mean_degree": float(degrees.mean()) if len(degrees) else 0.0,
        "components": len(components),
    }


# =============================================================================
# PART 6 — OBJECTIVE GRADIENT
# =============================================================================
def objective_gradient_for_robot(robot_index, positions, density, visible_indices):
    """Analytic gradient of the component-level detection objective w.r.t. r_i.

    p_i(q) = exp(-||q-r_i||^2 / (2 sigma^2))
    J = sum_q density(q) * [1 - product_j (1-p_j(q))]

    Therefore:
    dJ/dr_i = sum_q density(q) * p_i(q) * product_{j != i}(1-p_j(q))
                       * (q-r_i) / sigma^2.

    Only positions in the robot's current connected component are included.
    """
    visible_indices = list(visible_indices)
    visible_positions = np.asarray(positions, dtype=float)[visible_indices]
    local_index = visible_indices.index(robot_index)
    probs = per_robot_detection_probabilities(visible_positions)
    p_i = probs[local_index]

    if len(visible_indices) > 1:
        other_miss = np.prod(np.delete(1.0 - probs, local_index, axis=0), axis=0)
    else:
        other_miss = np.ones((GRID_SIZE, GRID_SIZE), dtype=float)

    rx, ry = positions[robot_index]
    weight = density * p_i * other_miss / (SIGMA_SENSOR**2)
    gx = float(np.sum(weight * (X - rx)))
    gy = float(np.sum(weight * (Y - ry)))
    return np.array([gx, gy], dtype=float)


def component_detection_score(positions, density, component_indices):
    """Detection score considering only robots in one communication component."""
    return detection_score(np.asarray(positions)[list(component_indices)], density)


def objective_gradient_controller(robot_positions, density, max_step):
    """Adaptive objective-gradient step with magnitude scaling and acceptance.

    Unlike the old normalized-gradient rule, which moved every robot by a full
    step whenever the gradient was nonzero, this version:
      1. Recomputes the communication graph as positions change.
      2. Uses each robot's current connected component for its gradient.
      3. Scales movement according to gradient magnitude (not always max_step).
      4. Rejects a proposed move if it decreases that component's current score.

    The acceptance check is local to the current density map at this time step.
    It does not guarantee improvement at future time steps because the target
    distribution itself moves.
    """
    updated = np.asarray(robot_positions, dtype=float).copy()
    accepted_steps = np.zeros(len(updated), dtype=float)

    for i in range(len(updated)):
        adjacency = communication_adjacency(updated, COMMUNICATION_RANGE)
        component = next(c for c in connected_components(adjacency) if i in c)
        gradient = objective_gradient_for_robot(i, updated, density, component)
        norm = float(np.linalg.norm(gradient))
        if norm <= GRADIENT_DEADBAND:
            continue

        direction = gradient / norm
        proposed_distance = min(max_step, OBJECTIVE_STEP_GAIN * norm)
        if proposed_distance <= 1e-12:
            continue

        baseline = component_detection_score(updated, density, component)
        distance = proposed_distance
        # Backtracking makes the controller conservative near a local maximum
        # or a workspace boundary. Try the proposed move, then halve it.
        for _ in range(12):
            candidate = updated.copy()
            candidate[i] = np.clip(
                updated[i] + distance * direction, 0.0, GRID_SIZE - 1.0
            )
            actual_distance = float(np.linalg.norm(candidate[i] - updated[i]))
            if actual_distance <= 1e-12:
                distance *= 0.5
                continue
            candidate_score = component_detection_score(candidate, density, component)
            if candidate_score >= baseline - 1e-12:
                updated = candidate
                accepted_steps[i] = actual_distance
                break
            distance *= 0.5

    return updated, accepted_steps


# =============================================================================
# PART 7 — DENSITY-GRADIENT BASELINE
# =============================================================================
def density_gradient_controller(robot_positions, density, max_step):
    """Move toward increasing local target density, using a bounded gradient step."""
    grad_y, grad_x = np.gradient(density)
    updated = np.asarray(robot_positions, dtype=float).copy()
    distances = np.zeros(len(updated), dtype=float)

    for i, (rx, ry) in enumerate(updated.copy()):
        ix = int(np.clip(round(rx), 0, GRID_SIZE - 1))
        iy = int(np.clip(round(ry), 0, GRID_SIZE - 1))
        gradient = np.array([grad_x[iy, ix], grad_y[iy, ix]], dtype=float)
        norm = float(np.linalg.norm(gradient))
        if norm <= GRADIENT_DEADBAND:
            continue

        # Preserve the original gradient-following baseline, but avoid moving
        # when the local density gradient is effectively zero.
        direction = gradient / norm
        candidate = np.clip(updated[i] + max_step * direction, 0, GRID_SIZE - 1)
        distances[i] = float(np.linalg.norm(candidate - updated[i]))
        updated[i] = candidate

    return updated, distances


# =============================================================================
# PART 8 — SIMULATION AND LOGGING
# =============================================================================
def run_simulation():
    rng = np.random.default_rng(RANDOM_SEED)
    uniform_prior = np.full((GRID_SIZE, GRID_SIZE), 1.0 / (GRID_SIZE * GRID_SIZE))

    # Controllers start without access to the hidden target-density map.
    random_robots = rng.uniform(0, GRID_SIZE - 1, size=(NUM_ROBOTS, 2))
    greedy_robots = greedy_initial_positions(uniform_prior, NUM_ROBOTS)

    # Fair initialization: both adaptive methods begin at the same greedy layout.
    density_robots = greedy_robots.copy()
    objective_robots = greedy_robots.copy()

    density_paths = [density_robots.copy()]
    objective_paths = [objective_robots.copy()]
    score_series = {
        "Random (static)": [],
        "Greedy (static)": [],
        "Density gradient": [],
        "Detection-objective gradient": [],
    }
    network_log = []
    travel_log = []
    movement_log = []
    density_travel = 0.0
    objective_travel = 0.0
    density_estimate = uniform_prior.copy()
    objective_density_estimate = uniform_prior.copy()
    estimate_paths = [objective_density_estimate.copy()]
    observation_log = []

    for t in range(TIME_STEPS):
        # Hidden truth is available to the simulator/evaluator, not to controllers.
        true_density = goal_density(t)
        observations = observe_targets(true_density, objective_robots, rng)
        observation_log.append({"time": t, "detected_targets": int(len(observations))})
        density_estimate = update_density_estimate(density_estimate, observations)
        objective_density_estimate = update_density_estimate(objective_density_estimate, observations)
        estimate_paths.append(objective_density_estimate.copy())
        density = true_density

        # Score all policies against the true density for fair evaluation only.
        score_series["Random (static)"].append(detection_score(random_robots, density))
        score_series["Greedy (static)"].append(detection_score(greedy_robots, density))
        score_series["Density gradient"].append(detection_score(density_robots, density))
        score_series["Detection-objective gradient"].append(
            detection_score(objective_robots, density)
        )

        adjacency = communication_adjacency(objective_robots, COMMUNICATION_RANGE)
        stats = network_statistics(adjacency)
        stats["time"] = t
        network_log.append(stats)

        old_density = density_robots.copy()
        density_robots, density_steps = density_gradient_controller(
            density_robots, density_estimate, MAX_STEP_DENSITY
        )
        density_step_distance = float(density_steps.sum())
        density_travel += density_step_distance
        density_paths.append(density_robots.copy())

        old_objective = objective_robots.copy()
        objective_robots, objective_steps = objective_gradient_controller(
            objective_robots, objective_density_estimate, MAX_STEP_OBJECTIVE
        )
        objective_step_distance = float(objective_steps.sum())
        objective_travel += objective_step_distance
        objective_paths.append(objective_robots.copy())

        movement_log.append({
            "time": t,
            "density_gradient_step_travel": density_step_distance,
            "objective_gradient_step_travel": objective_step_distance,
            "objective_robots_moved": int(np.count_nonzero(objective_steps > 1e-9)),
            "objective_mean_robot_step": float(objective_steps.mean()),
            "objective_score_before_move": score_series["Detection-objective gradient"][-1],
            "objective_score_after_move_same_density": detection_score(objective_robots, density),
        })
        travel_log.append({
            "time": t,
            "density_gradient_total_travel": density_travel,
            "objective_gradient_total_travel": objective_travel,
        })

    # CSV outputs for reproducibility and subsequent analysis.
    with (OUTPUT_DIR / "detection_scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        names = list(score_series.keys())
        writer.writerow(["time"] + names)
        for t in range(TIME_STEPS):
            writer.writerow([t] + [score_series[name][t] for name in names])

    with (OUTPUT_DIR / "communication_network.csv").open("w", newline="") as f:
        fields = ["time", "connected", "edges", "mean_degree", "components"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(network_log)

    with (OUTPUT_DIR / "travel_distance.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "time", "density_gradient_total_travel", "objective_gradient_total_travel"
        ])
        writer.writeheader()
        writer.writerows(travel_log)

    with (OUTPUT_DIR / "movement_diagnostics.csv").open("w", newline="") as f:
        fields = list(movement_log[0].keys()) if movement_log else []
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(movement_log)

    with (OUTPUT_DIR / "observation_counts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "detected_targets"])
        writer.writeheader()
        writer.writerows(observation_log)

    print("\nAVERAGE DETECTION SCORE (score before movement at each time step)")
    for name, values in score_series.items():
        print(f"{name:36s}: {np.mean(values):.6f}")

    print("\nCOMMUNICATION NETWORK SUMMARY")
    print(f"Communication range: {COMMUNICATION_RANGE:.2f}")
    print(f"Fraction of time steps connected: {np.mean([r['connected'] for r in network_log]):.3f}")
    print(f"Mean number of links: {np.mean([r['edges'] for r in network_log]):.2f}")
    print(f"Mean node degree: {np.mean([r['mean_degree'] for r in network_log]):.2f}")

    print("\nTOTAL TRAVEL DISTANCE")
    print(f"Density-gradient controller: {density_travel:.3f}")
    print(f"Detection-objective gradient controller: {objective_travel:.3f}")
    print("\nMovement diagnostics:")
    print(f"Mean robots moving per step: {np.mean([r['objective_robots_moved'] for r in movement_log]):.2f} / {NUM_ROBOTS}")
    print(f"Mean objective-controller robot step: {np.mean([r['objective_mean_robot_step'] for r in movement_log]):.4f}")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")

    return {
        "scores": score_series,
        "random_robots": random_robots,
        "greedy_robots": greedy_robots,
        "density_paths": np.asarray(density_paths),
        "objective_paths": np.asarray(objective_paths),
        "objective_final_positions": objective_robots.copy(),
        "final_density_estimate": objective_density_estimate.copy(),
        "density_estimate_paths": np.asarray(estimate_paths),
        "observation_log": observation_log,
        "network_log": network_log,
        "movement_log": movement_log,
    }


# =============================================================================
# PART 9 — VISUALIZATION
# =============================================================================
def plot_results(results):
    scores = results["scores"]

    # Figure 1: detection score as the moving target-density field evolves.
    plt.figure(figsize=(11, 5))
    for label, values in scores.items():
        plt.plot(values, label=label, linewidth=1.5)
    plt.xlabel("Time step")
    plt.ylabel("Expected detection probability")
    plt.title("Detection Performance under a Moving Target-Density Field")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "detection_performance.png", dpi=200)
    plt.show()

    # Figure 2: adaptive trajectories on the final target-density map.
    final_density = goal_density(TIME_STEPS - 1)
    plt.figure(figsize=(9, 8))
    plt.imshow(final_density, origin="lower", extent=[0, GRID_SIZE - 1, 0, GRID_SIZE - 1])
    plt.colorbar(label="Target probability mass")
    plt.scatter(results["random_robots"][:, 0], results["random_robots"][:, 1],
                marker="x", s=90, label="Random static")
    plt.scatter(results["greedy_robots"][:, 0], results["greedy_robots"][:, 1],
                marker="s", s=65, label="Greedy static")

    density_paths = results["density_paths"]
    objective_paths = results["objective_paths"]
    for i in range(NUM_ROBOTS):
        plt.plot(density_paths[:, i, 0], density_paths[:, i, 1], "--", alpha=0.8,
                 label="Density-gradient path" if i == 0 else None)
        plt.plot(objective_paths[:, i, 0], objective_paths[:, i, 1], "-", alpha=0.9,
                 label="Objective-gradient path" if i == 0 else None)
        plt.scatter(objective_paths[-1, i, 0], objective_paths[-1, i, 1], marker="*", s=130)
    plt.xlim(0, GRID_SIZE - 1)
    plt.ylim(0, GRID_SIZE - 1)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Robot Trajectories on the Final Moving-Density Snapshot")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "robot_trajectories.png", dpi=200)
    plt.show()

    # Figure 3: final combined sensor-detection probability.
    final_positions = results["objective_final_positions"]
    coverage = sensor_detection(final_positions)
    plt.figure(figsize=(8, 7))
    plt.imshow(coverage, origin="lower", extent=[0, GRID_SIZE - 1, 0, GRID_SIZE - 1],
               vmin=0, vmax=1)
    plt.contour(X, Y, coverage, levels=[0.05, 0.25, 0.50, 0.70, 0.90],
                colors="white", linewidths=0.8)
    plt.scatter(final_positions[:, 0], final_positions[:, 1], marker="*", s=150,
                edgecolors="black", label="Robots")
    plt.colorbar(label="Combined detection probability")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Final Combined Probabilistic Detection Map")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_coverage.png", dpi=200)
    plt.show()

    # Figure 3b: proposed-method coverage aligned with target concentration.
    # The color map is team detection probability; dashed contours show the
    # target-density distribution at the same final simulation time.
    final_density = goal_density(TIME_STEPS - 1)
    plt.figure(figsize=(8, 7))
    plt.imshow(coverage, origin="lower", extent=[0, GRID_SIZE - 1, 0, GRID_SIZE - 1],
               vmin=0, vmax=1, cmap="viridis")
    density_levels = np.linspace(float(final_density.max()) * 0.2,
                                 float(final_density.max()) * 0.8, 4)
    density_contours = plt.contour(X, Y, final_density, levels=density_levels,
                                   colors="white", linestyles="--", linewidths=1.2)
    plt.clabel(density_contours, inline=True, fontsize=8, fmt="%.3g")
    plt.scatter(final_positions[:, 0], final_positions[:, 1], marker="*", s=170,
                edgecolors="black", linewidths=0.8, label="Proposed-method robots")
    plt.colorbar(label="Combined detection probability")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Proposed Method: Coverage Aligned with Target Density")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "proposed_method_coverage.png", dpi=220)
    plt.show()

    # Figure 3c: estimated target density available to the controller.
    estimate = results["final_density_estimate"]
    plt.figure(figsize=(8, 7))
    plt.imshow(estimate, origin="lower", extent=[0, GRID_SIZE - 1, 0, GRID_SIZE - 1])
    plt.scatter(final_positions[:, 0], final_positions[:, 1], marker="*", s=150,
                edgecolors="black", label="Robots")
    plt.colorbar(label="Estimated target probability mass")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Final Estimated Target Density (Observations Only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "estimated_target_density.png", dpi=220)
    plt.show()

    # Figure 4: cumulative travel distance.
    travel_path = OUTPUT_DIR / "travel_distance.csv"
    if travel_path.exists():
        with travel_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        t = [int(row["time"]) for row in rows]
        d = [float(row["density_gradient_total_travel"]) for row in rows]
        o = [float(row["objective_gradient_total_travel"]) for row in rows]
        plt.figure(figsize=(10, 5))
        plt.plot(t, d, label="Density gradient")
        plt.plot(t, o, label="Detection-objective gradient")
        plt.xlabel("Time step")
        plt.ylabel("Cumulative robot travel distance")
        plt.title("Cumulative Travel Comparison")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "travel_comparison.png", dpi=200)
        plt.show()

    # Figure 5: final peer-to-peer graph.
    adjacency = communication_adjacency(final_positions, COMMUNICATION_RANGE)
    plt.figure(figsize=(8, 7))
    plt.scatter(final_positions[:, 0], final_positions[:, 1], s=150)
    for i, (rx, ry) in enumerate(final_positions):
        plt.text(rx + 0.5, ry + 0.5, f"R{i}", fontsize=10)
    for i in range(NUM_ROBOTS):
        for j in range(i + 1, NUM_ROBOTS):
            if adjacency[i, j]:
                plt.plot([final_positions[i, 0], final_positions[j, 0]],
                         [final_positions[i, 1], final_positions[j, 1]])
    plt.xlim(0, GRID_SIZE - 1)
    plt.ylim(0, GRID_SIZE - 1)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Final Peer-to-Peer Communication Graph")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "communication_graph.png", dpi=200)
    plt.show()


# =============================================================================
# PART 10 — MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    simulation_results = run_simulation()
    plot_results(simulation_results)
