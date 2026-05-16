# Ordered Baseline Comparison Tables

Source: `classifier/tt_stratified_rerun.csv`.

Metrics are weighted precision, weighted recall, and weighted F1.
Hierarchical VAE, random sampling, and flat VAE are compared below.

## TrainTicket

| Real data | Synthetic data | Hierarchical VAE precision | Hierarchical VAE recall | Hierarchical VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100% | 0.410 | 0.446 | 0.421 |  |  |  |  |  |  |
| 10% | 90% | 0.659 | 0.656 | 0.627 |  |  |  |  |  |  |
| 30% | 70% | 0.726 | 0.718 | 0.701 |  |  |  |  |  |  |
| 50% | 50% | 0.794 | 0.717 | 0.713 |  |  |  |  |  |  |
| 70% | 30% | 0.830 | 0.727 | 0.720 |  |  |  |  |  |  |
| 100% | 0% | 0.836 | 0.728 | 0.720 |  |  |  |  |  |  |
