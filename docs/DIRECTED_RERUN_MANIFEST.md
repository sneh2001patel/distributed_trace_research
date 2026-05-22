# Directed Rerun Manifest

This manifest records the final directed rerun artifacts used for the paper results.

## Included result files

### Tables

- `results/icsme2026_final/tables/structural_fidelity_directed_full_all.csv`
- `results/icsme2026_final/tables/structural_fidelity_directed_full_all.md`
- `results/icsme2026_final/tables/memorization_directed_full.csv`
- `results/icsme2026_final/tables/filtering_sensitivity.csv`
- `results/icsme2026_final/tables/baseline_directed_full_hierarchical.csv`
- `results/icsme2026_final/tables/baseline_directed_full_hierarchical.md`
- `results/icsme2026_final/tables/baseline_directed_full_extended.csv`
- `results/icsme2026_final/tables/baseline_directed_full_extended.md`
- `results/icsme2026_final/tables/hierarchical_repeated_seed_summary.csv`
- `results/icsme2026_final/tables/hierarchical_repeated_seeds.csv`
- `results/icsme2026_final/tables/hierarchical_repeated_seeds.md`

### Logs

- `results/icsme2026_final/logs/full_directed_generation.log`
- `results/icsme2026_final/logs/generate_directed_baselines.log`
- `results/icsme2026_final/logs/structural_fidelity_directed_full_all.log`
- `results/icsme2026_final/logs/baseline_directed_full_hierarchical.log`
- `results/icsme2026_final/logs/baseline_directed_full_random_flat.log`
- `results/icsme2026_final/logs/hierarchical_repeated_seeds.log`
- `results/icsme2026_final/logs/extra_analyses.log`

### Dataset summaries

- `results/icsme2026_final/summaries/TT_synthetic_summary.json`
- `results/icsme2026_final/summaries/SN_synthetic_summary.json`
- `results/icsme2026_final/summaries/random_sampling_summary.json`

## Notes

- These artifacts supersede older undirected and smoke-test outputs.
- The final graph generation preserves directed parent-child relationships.
- The downstream classifier results include weighted F1, PR-AUC, false-alarm rate, and confusion-count diagnostics where available.
