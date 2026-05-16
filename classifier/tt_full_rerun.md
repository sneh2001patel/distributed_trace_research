# Ordered Baseline Comparison Tables

Source: `classifier/tt_full_rerun.csv`.

Metrics are weighted precision, weighted recall, and weighted F1.
Hierarchical VAE, random sampling, and flat VAE are compared below.

## TrainTicket

| Real data | Synthetic data | Hierarchical VAE precision | Hierarchical VAE recall | Hierarchical VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100% | 0.619 | 0.604 | 0.607 | 0.758 | 0.586 | 0.435 | 0.679 | 0.606 | 0.490 |
| 10% | 90% | 0.706 | 0.701 | 0.702 | 0.834 | 0.728 | 0.720 | 0.675 | 0.679 | 0.674 |
| 30% | 70% | 0.781 | 0.723 | 0.721 | 0.834 | 0.727 | 0.719 | 0.794 | 0.737 | 0.735 |
| 50% | 50% | 0.772 | 0.727 | 0.727 | 0.836 | 0.728 | 0.720 | 0.788 | 0.735 | 0.734 |
| 70% | 30% | 0.738 | 0.723 | 0.725 | 0.836 | 0.728 | 0.720 | 0.769 | 0.732 | 0.733 |
| 100% | 0% | 0.836 | 0.728 | 0.720 | 0.836 | 0.728 | 0.720 | 0.836 | 0.728 | 0.720 |
