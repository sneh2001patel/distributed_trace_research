# Result Provenance

This document maps the main paper claims to the final result files included in this repository. The final paper uses the directed rerun artifacts under `results/icsme2026_final/`.

## Canonical result root

```text
results/icsme2026_final/
```

## RQ2: Generated-trace fidelity

Source files:

- `results/icsme2026_final/tables/structural_fidelity_directed_full_all.csv`
- `results/icsme2026_final/tables/structural_fidelity_directed_full_all.md`

Use the rows where `trace_class=combined`.

Paper values:

| System | Generator | Structure | Duration | Semantic | Overall |
|---|---|---:|---:|---:|---:|
| TrainTicket | Random Sampling | 0.568 | 0.829 | 0.330 | 0.575 |
| TrainTicket | Flat VAE | 0.601 | 0.400 | 0.632 | 0.545 |
| TrainTicket | Hierarchical VAE | 0.982 | 0.952 | 0.986 | 0.974 |
| SocialNetwork | Random Sampling | 0.795 | 0.853 | 0.387 | 0.678 |
| SocialNetwork | Flat VAE | 0.639 | 0.429 | 0.761 | 0.610 |
| SocialNetwork | Hierarchical VAE | 0.624 | 0.402 | 0.526 | 0.517 |

The overall score is the mean of structure, duration, and semantic means.

## RQ2: Memorization and novelty

Source file:

- `results/icsme2026_final/tables/memorization_directed_full.csv`

Paper values:

| System | Class | Unique | Exact duplicate | Novel | Near duplicate |
|---|---|---:|---:|---:|---:|
| TrainTicket | Normal | 0.012 | 0.598 | 0.402 | 0.988 |
| TrainTicket | Abnormal | 0.012 | 0.563 | 0.437 | 0.994 |
| SocialNetwork | Normal | 0.963 | 0.000 | 1.000 | 0.000 |
| SocialNetwork | Abnormal | 0.815 | 0.000 | 1.000 | 0.005 |

Interpretation: TrainTicket has high fidelity but high near-duplicate risk. SocialNetwork has greater novelty but weaker aggregate alignment.

## RQ3: Mixed real/synthetic anomaly detection

Source files:

- `results/icsme2026_final/tables/baseline_directed_full_hierarchical.csv`
- `results/icsme2026_final/tables/baseline_directed_full_extended.csv`

Paper values:

| System | Generator | 0% | 10% | 30% | 50% | 70% |
|---|---|---:|---:|---:|---:|---:|
| TrainTicket | Hierarchical VAE | 0.630 | 0.636 | 0.635 | 0.648 | 0.662 |
| TrainTicket | Random Sampling | 0.544 | 0.566 | 0.571 | 0.657 | 0.662 |
| TrainTicket | Flat VAE | 0.490 | 0.625 | 0.659 | 0.663 | 0.674 |
| SocialNetwork | Hierarchical VAE | 0.479 | 0.637 | 0.655 | 0.658 | 0.655 |
| SocialNetwork | Random Sampling | 0.479 | 0.657 | 0.650 | 0.684 | 0.654 |
| SocialNetwork | Flat VAE | 0.206 | 0.613 | 0.652 | 0.664 | 0.659 |

The table reports weighted F1 on held-out real test traces.

## RQ3: Repeated-seed hierarchical VAE

Source files:

- `results/icsme2026_final/tables/hierarchical_repeated_seed_summary.csv`
- `results/icsme2026_final/tables/hierarchical_repeated_seeds.csv`

Paper values:

| System | 0% real | 10% real | 30% real | 100% real |
|---|---:|---:|---:|---:|
| TrainTicket | 0.638 +/- 0.011 | 0.643 +/- 0.012 | 0.648 +/- 0.010 | 0.675 +/- 0.011 |
| SocialNetwork | 0.479 +/- 0.000 | 0.638 +/- 0.009 | 0.654 +/- 0.007 | 0.663 +/- 0.005 |

## False-alarm diagnostics

Source file:

- `results/icsme2026_final/tables/hierarchical_repeated_seed_summary.csv`

Paper values:

| System | 0% real | 30% real | 100% real |
|---|---:|---:|---:|
| TrainTicket | 0.311 +/- 0.016 | 0.275 +/- 0.057 | 0.106 +/- 0.012 |
| SocialNetwork | 1.000 +/- 0.000 | 0.568 +/- 0.072 | 0.559 +/- 0.021 |

## Filtering sensitivity

Source file:

- `results/icsme2026_final/tables/filtering_sensitivity.csv`

The saved directed datasets retain all evaluated graphs at both 30-span and 50-span caps.

## Implementation notes

- Directed parent-child orientation is preserved in the final synthetic graph construction.
- Generation samples graph and node latents from a training latent pool with noise.
- Held-out real test traces are not used to build the generation pool.
- The downstream classifier is a two-layer GCN trained with class-weighted cross entropy.
- The reported downstream metrics use held-out real test traces.
