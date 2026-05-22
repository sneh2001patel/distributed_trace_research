# Synthetic Distributed Traces for Industrial Observability

This repository contains the code and final result artifacts for the paper:

**Reducing Trace Data Scarcity in Industrial Observability Pipelines Using Synthetic Distributed Traces**

The study evaluates whether synthetic distributed traces can help when production telemetry is limited, sensitive, or difficult to share. The experiments use the TrainTicket and SocialNetwork microservice benchmarks and compare hierarchical VAE generation with simpler baselines.

## Repository layout

```text
src/                         Core preprocessing, generation, evaluation, and classifier code
src/classifier/              GCN anomaly-detection components
results/icsme2026_final/     Final directed rerun tables, logs, and dataset summaries
docs/                        Result provenance and paper-result mapping
data/                        Dataset acquisition notes
```

## Main results included here

- Directed synthetic trace generation for TrainTicket and SocialNetwork.
- Structural, duration, semantic, and aggregate fidelity comparisons.
- Memorization and novelty diagnostics.
- Mixed real/synthetic anomaly-detection results.
- Five-seed hierarchical VAE downstream summaries.
- Filtering sensitivity checks.

The included results correspond to the final directed rerun used for the paper tables.

## Important interpretation

Synthetic traces are evaluated as lower-exposure bootstrap and augmentation data. They are not a formal privacy guarantee. The results show a workload-dependent tradeoff: TrainTicket generations preserve structure well but have higher near-duplicate risk, while SocialNetwork generations are more novel but less aligned on aggregate similarity.

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the expected workflow and [docs/RESULT_PROVENANCE.md](docs/RESULT_PROVENANCE.md) for the mapping between paper claims and result files.
