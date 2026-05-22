# Ordered Baseline Comparison Tables

Source: `classifier/baseline_directed_full_hierarchical.csv`.

Metrics are weighted precision, weighted recall, and weighted F1.
Hierarchical VAE, random sampling, and flat VAE are compared below.

## TrainTicket

| Real data | Synthetic data | Hierarchical VAE precision | Hierarchical VAE recall | Hierarchical VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100% | 0.630 | 0.631 | 0.630 | 0.646 | 0.622 | 0.544 | 0.622 | 0.600 | 0.490 |
| 10% | 90% | 0.637 | 0.635 | 0.636 | 0.566 | 0.578 | 0.566 | 0.688 | 0.666 | 0.625 |
| 30% | 70% | 0.637 | 0.633 | 0.635 | 0.638 | 0.628 | 0.571 | 0.663 | 0.668 | 0.659 |
| 100% | 0% | 0.692 | 0.685 | 0.663 | 0.696 | 0.692 | 0.672 | 0.692 | 0.684 | 0.660 |

## SocialNetwork

| Real data | Synthetic data | Hierarchical VAE precision | Hierarchical VAE recall | Hierarchical VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100% | 0.389 | 0.623 | 0.479 | 0.389 | 0.624 | 0.479 | 0.142 | 0.376 | 0.206 |
| 10% | 90% | 0.635 | 0.648 | 0.637 | 0.655 | 0.663 | 0.657 | 0.635 | 0.607 | 0.613 |
| 30% | 70% | 0.654 | 0.665 | 0.655 | 0.673 | 0.681 | 0.650 | 0.659 | 0.671 | 0.652 |
| 100% | 0% | 0.665 | 0.671 | 0.667 | 0.661 | 0.671 | 0.661 | 0.660 | 0.668 | 0.662 |
