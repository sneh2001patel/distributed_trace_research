# Full Rerun Baseline Comparison Tables

Source: `classifier/baseline_comparison_full_rerun_results.csv`. Metrics are weighted precision, weighted recall, and weighted F1.

## TrainTicket

| Real data | Synthetic data | Hierarchial VAE precision | Hierarchial VAE recall | Hierarchial VAE F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100% | 0.506 | 0.531 | 0.508 | 0.645 | 0.588 | 0.444 | 0.757 | 0.585 | 0.433 |
| 10% | 90% | 0.746 | 0.712 | 0.712 | 0.734 | 0.702 | 0.703 | 0.805 | 0.716 | 0.710 |
| 30% | 70% | 0.754 | 0.724 | 0.725 | 0.822 | 0.725 | 0.718 | 0.823 | 0.722 | 0.715 |
| 50% | 50% | 0.787 | 0.732 | 0.730 | 0.775 | 0.734 | 0.734 | 0.834 | 0.727 | 0.719 |
| 70% | 30% | 0.836 | 0.728 | 0.720 | 0.775 | 0.735 | 0.736 | 0.836 | 0.728 | 0.720 |
| 100% | 0% | 0.836 | 0.728 | 0.720 | 0.836 | 0.728 | 0.720 | 0.836 | 0.728 | 0.720 |

## SocialNetwork

| Real data | Synthetic data | Hierarchial VAE precision | Hierarchial VAE recall | Hierarchial VAE F1 score | Flat VAE precision | Flat VAE recall | Flat VAE F1 score | Random sampling precision | Random sampling recall | Random sampling F1 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100% | 0.651 | 0.578 | 0.599 | 0.522 | 0.722 | 0.606 | 0.077 | 0.277 | 0.120 |
| 10% | 90% | 0.787 | 0.791 | 0.788 | 0.802 | 0.791 | 0.795 | 0.849 | 0.827 | 0.832 |
| 30% | 70% | 0.815 | 0.820 | 0.816 | 0.849 | 0.819 | 0.826 | 0.854 | 0.838 | 0.843 |
| 50% | 50% | 0.874 | 0.846 | 0.852 | 0.861 | 0.843 | 0.847 | 0.861 | 0.848 | 0.852 |
| 70% | 30% | 0.877 | 0.844 | 0.851 | 0.870 | 0.848 | 0.853 | 0.872 | 0.862 | 0.865 |
| 100% | 0% | 0.901 | 0.853 | 0.860 | 0.900 | 0.853 | 0.860 | 0.900 | 0.853 | 0.860 |
