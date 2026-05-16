# Ordered Baseline Comparison Tables

Source: `classifier/tt_hier_fixed_generation_results.csv`.

Metrics are weighted precision, weighted recall, and weighted F1.
Hierarchical VAE, random sampling, and flat VAE are compared below.

## TrainTicket

| Real data | Synthetic data | Hierarchical VAE precision | Hierarchical VAE recall | Hierarchical VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100% | 0.690 | 0.688 | 0.689 |  |  |  |  |  |  |
| 10% | 90% | 0.689 | 0.690 | 0.690 |  |  |  |  |  |  |
| 30% | 70% | 0.718 | 0.718 | 0.718 |  |  |  |  |  |  |
| 50% | 50% | 0.762 | 0.737 | 0.738 |  |  |  |  |  |  |
| 70% | 30% | 0.752 | 0.722 | 0.724 |  |  |  |  |  |  |
| 100% | 0% | 0.836 | 0.728 | 0.720 |  |  |  |  |  |  |
