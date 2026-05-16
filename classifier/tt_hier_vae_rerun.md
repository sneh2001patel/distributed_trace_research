# Ordered Baseline Comparison Tables

Source: `classifier/tt_hier_vae_rerun.csv`.

Metrics are weighted precision, weighted recall, and weighted F1.
Hierarchical VAE, random sampling, and flat VAE are compared below.

## TrainTicket

| Real data | Synthetic data | Hierarchical VAE precision | Hierarchical VAE recall | Hierarchical VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100% | 0.620 | 0.608 | 0.610 |  |  |  |  |  |  |
| 10% | 90% | 0.708 | 0.701 | 0.703 |  |  |  |  |  |  |
| 30% | 70% | 0.795 | 0.725 | 0.721 |  |  |  |  |  |  |
| 50% | 50% | 0.767 | 0.727 | 0.727 |  |  |  |  |  |  |
| 70% | 30% | 0.787 | 0.719 | 0.716 |  |  |  |  |  |  |
| 100% | 0% | 0.836 | 0.728 | 0.720 |  |  |  |  |  |  |
