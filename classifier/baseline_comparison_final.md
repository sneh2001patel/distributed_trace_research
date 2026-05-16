# Ordered Baseline Comparison Tables

Source: `classifier/baseline_comparison_final.csv`.

Metrics are weighted precision, weighted recall, and weighted F1.
Hierarchical VAE, random sampling, and flat VAE are compared below.

## TrainTicket

| Real data | Synthetic data | Hierarchical VAE precision | Hierarchical VAE recall | Hierarchical VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100% | 0.668 | 0.651 | 0.605 | 0.758 | 0.586 | 0.435 | 0.679 | 0.606 | 0.490 |
| 10% | 90% | 0.675 | 0.670 | 0.643 | 0.833 | 0.727 | 0.719 | 0.676 | 0.681 | 0.675 |
| 30% | 70% | 0.734 | 0.728 | 0.730 | 0.836 | 0.728 | 0.720 | 0.799 | 0.728 | 0.725 |
| 50% | 50% | 0.765 | 0.735 | 0.737 | 0.836 | 0.728 | 0.720 | 0.781 | 0.730 | 0.729 |
| 70% | 30% | 0.749 | 0.724 | 0.725 | 0.836 | 0.728 | 0.720 | 0.758 | 0.734 | 0.735 |
| 100% | 0% | 0.836 | 0.728 | 0.720 | 0.836 | 0.728 | 0.720 | 0.836 | 0.728 | 0.720 |

## SocialNetwork

| Real data | Synthetic data | Hierarchical VAE precision | Hierarchical VAE recall | Hierarchical VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100% | 0.799 | 0.766 | 0.775 | 0.077 | 0.277 | 0.120 | 0.539 | 0.720 | 0.605 |
| 10% | 90% | 0.823 | 0.823 | 0.823 | 0.835 | 0.829 | 0.831 | 0.831 | 0.820 | 0.823 |
| 30% | 70% | 0.859 | 0.846 | 0.850 | 0.861 | 0.848 | 0.852 | 0.858 | 0.842 | 0.846 |
| 50% | 50% | 0.878 | 0.853 | 0.858 | 0.875 | 0.860 | 0.864 | 0.875 | 0.849 | 0.854 |
| 70% | 30% | 0.872 | 0.847 | 0.853 | 0.874 | 0.862 | 0.865 | 0.876 | 0.853 | 0.858 |
| 100% | 0% | 0.900 | 0.853 | 0.860 | 0.899 | 0.852 | 0.859 | 0.900 | 0.853 | 0.860 |
