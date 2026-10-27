# unknown_density_multiseed_validation_bundle
## License
Copyright © 2026 **Md Shaharia Hossen**.

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for the full license text.

The MIT License permits use, copying, modification, merging, publishing, distribution, sublicensing, and sale of copies of the software, subject to the license conditions.



# Online Density Estimation + Multi-Seed Validation

## 1. Install dependencies
```bash
python -m pip install numpy scipy pandas matplotlib
```

## 2. Run 10 seeds
Keep `adaptive_multirobot_networked_coverage_revised.py` in the same directory as the runner, then run:
```bash
python run_multiseed_validation.py
```

## 3. Run a larger experiment
```bash
python run_multiseed_validation.py --seeds 30 --start-seed 100 --output-dir results_30_seeds
```

To preserve the model's own per-seed CSVs and plots:
```bash
python run_multiseed_validation.py --seeds 10 --keep-seed-outputs
```

If the model source is in another location:
```bash
python run_multiseed_validation.py --model /path/to/adaptive_multirobot_networked_coverage_revised.py --seeds 10
```

## Outputs
- `density_metrics_by_seed_time.csv`: all density metrics for every seed and time step.
- `density_metrics_over_time_mean_std.csv`: mean and standard deviation over seeds for each time step.
- `density_metrics_summary_by_seed.csv`: per-seed average density metrics.
- `detection_scores_by_seed_time.csv`: each method's detection score per seed/time.
- `detection_scores_over_time_mean_std.csv`: mean and standard deviation over seeds per time and method.
- `per_seed_mean_scores.csv`: time-averaged detection score for each method and seed.
- `detection_scores_across_seeds_mean_std.csv`: overall mean and sample SD across seeds.
- PNG plots for RMSE, JS divergence, centroid error, and detection performance.
- `run_metadata.json`: settings and metric interpretation.

## Evaluation details
At time t, the model first observes sampled targets and updates its online density estimate.
The runner compares that updated estimate (`density_estimate_paths[t+1]`) with the hidden
true map (`goal_density(t)`). The model's controllers use observations and estimated density;
the true density is used by the model for evaluation of detection scores and by this runner
for the density-estimation metrics. The runner does not change the control algorithms.
