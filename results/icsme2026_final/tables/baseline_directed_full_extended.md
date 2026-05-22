# Ordered Baseline Comparison Tables

Source: `classifier/baseline_directed_full_extended.csv`.

Metrics are weighted precision, weighted recall, and weighted F1.
Hierarchical VAE, random sampling, and flat VAE are compared below.

## TrainTicket

| Real data | Synthetic data | Hierarchical VAE precision | Hierarchical VAE recall | Hierarchical VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | 50% | 0.651 | 0.657 | 0.648 | 0.666 | 0.670 | 0.657 | 0.676 | 0.678 | 0.663 |
| 70% | 30% | 0.679 | 0.679 | 0.662 | 0.687 | 0.683 | 0.662 | 0.696 | 0.692 | 0.674 |

## SocialNetwork

| Real data | Synthetic data | Hierarchical VAE precision | Hierarchical VAE recall | Hierarchical VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | 50% | 0.663 | 0.675 | 0.658 | 0.683 | 0.690 | 0.684 | 0.665 | 0.676 | 0.664 |
| 70% | 30% | 0.653 | 0.662 | 0.655 | 0.653 | 0.655 | 0.654 | 0.657 | 0.663 | 0.659 |
