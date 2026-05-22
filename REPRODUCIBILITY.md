# Reproducibility Notes

## Environment

The experiments were run with Python, PyTorch, PyTorch Geometric, NumPy, pandas, scikit-learn, and NetworkX. GPU execution is recommended for retraining the VAE models and rerunning the downstream classifier sweeps.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## High-level workflow

The final paper results were produced with directed graph generation. Older undirected outputs and smoke-test outputs should not be used for paper claims.

Typical workflow:

```bash
python src/build_anomaly_trace_graphs.py
python src/generate_tt_synthetic_data.py
python src/generate_sn_synthetic_data.py
python src/random_sampling_baselines.py
python src/generate_flat_synthetic_data.py
python src/structural_fidelity.py
python src/analyze_memorization.py
python src/analyze_filtering.py
python src/run_baseline_comparison.py
python src/aggregate_seed_results.py
```

The scripts expect the benchmark traces, preprocessed graph files, and trained weights to be available in the same directory conventions used by the original experiment workspace. Large raw traces and model weights are not bundled in this cleaned repository.

## Final artifact location

The final tables are under:

```text
results/icsme2026_final/tables/
```

The run logs are under:

```text
results/icsme2026_final/logs/
```

Dataset summary JSON files are under:

```text
results/icsme2026_final/summaries/
```
