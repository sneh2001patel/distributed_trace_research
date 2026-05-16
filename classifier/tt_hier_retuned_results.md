# Ordered Baseline Comparison Tables

Source: `classifier/tt_hier_retuned_results.csv`.

Metrics are weighted precision, weighted recall, and weighted F1.
Hierarchical VAE, random sampling, and flat VAE are compared below.

## TrainTicket

| Real data | Synthetic data | Hierarchical VAE precision | Hierarchical VAE recall | Hierarchical VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100% | 0.726 | 0.728 | 0.725 |  |  |  |  |  |  |
| 10% | 90% | 0.720 | 0.710 | 0.712 |  |  |  |  |  |  |
| 30% | 70% | 0.774 | 0.733 | 0.733 |  |  |  |  |  |  |
| 50% | 50% | 0.763 | 0.732 | 0.733 |  |  |  |  |  |  |
| 70% | 30% | 0.785 | 0.737 | 0.736 |  |  |  |  |  |  |
| 100% | 0% | 0.836 | 0.728 | 0.720 |  |  |  |  |  |  |
