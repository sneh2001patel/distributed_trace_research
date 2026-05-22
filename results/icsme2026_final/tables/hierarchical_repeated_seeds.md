# Ordered Baseline Comparison Tables

Source: `classifier/hierarchical_repeated_seeds.csv`.

Metrics are weighted precision, weighted recall, and weighted F1.
Hierarchical VAE, random sampling, and flat VAE are compared below.

## TrainTicket

| Real data | Synthetic data | Hierarchical VAE precision | Hierarchical VAE recall | Hierarchical VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100% | 0.655 | 0.653 | 0.654 |  |  |  |  |  |  |
| 10% | 90% | 0.648 | 0.644 | 0.645 |  |  |  |  |  |  |
| 30% | 70% | 0.665 | 0.669 | 0.657 |  |  |  |  |  |  |
| 100% | 0% | 0.694 | 0.688 | 0.668 |  |  |  |  |  |  |

## SocialNetwork

| Real data | Synthetic data | Hierarchical VAE precision | Hierarchical VAE recall | Hierarchical VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100% | 0.389 | 0.624 | 0.479 |  |  |  |  |  |  |
| 10% | 90% | 0.622 | 0.628 | 0.625 |  |  |  |  |  |  |
| 30% | 70% | 0.656 | 0.670 | 0.647 |  |  |  |  |  |  |
| 100% | 0% | 0.662 | 0.674 | 0.658 |  |  |  |  |  |  |
