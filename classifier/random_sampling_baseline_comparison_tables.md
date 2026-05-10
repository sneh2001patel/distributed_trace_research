# Random Sampling Baseline Comparison Results

Filtered from `classifier/baseline_comparison_results.csv` for `generator = random_sampling` only. If duplicate runs exist, this table uses the latest CSV row for each system and real-data percentage.

Weighted metrics can look better than the classifier really is when one class dominates. Balanced accuracy and macro F1 are included to expose single-class collapse.

## TrainTicket

| Real % | Synthetic % | Weighted Precision | Weighted Recall | Weighted F1 | Balanced Acc. | Macro F1 | Normal F1 | Abnormal F1 | Test Acc. | CSV Line |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100 | 0.342 | 0.585 | 0.431 | 0.500 | 0.369 | 0.738 | 0.000 | 0.585 | 63 |
| 10 | 90 | 0.809 | 0.718 | 0.712 | 0.754 | 0.716 | 0.691 | 0.740 | 0.718 | 64 |
| 30 | 70 | 0.831 | 0.726 | 0.719 | 0.765 | 0.723 | 0.696 | 0.751 | 0.726 | 65 |
| 50 | 50 | 0.834 | 0.727 | 0.719 | 0.766 | 0.724 | 0.696 | 0.752 | 0.727 | 66 |
| 70 | 30 | 0.833 | 0.727 | 0.719 | 0.766 | 0.724 | 0.696 | 0.752 | 0.727 | 67 |
| 100 | 0 | 0.836 | 0.728 | 0.720 | 0.767 | 0.725 | 0.696 | 0.753 | 0.728 | 68 |

## SocialNetwork

| Real % | Synthetic % | Weighted Precision | Weighted Recall | Weighted F1 | Balanced Acc. | Macro F1 | Normal F1 | Abnormal F1 | Test Acc. | CSV Line |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100 | 0.077 | 0.277 | 0.120 | 0.500 | 0.217 | 0.000 | 0.434 | 0.277 | 69 |
| 10 | 90 | 0.844 | 0.825 | 0.830 | 0.820 | 0.796 | 0.872 | 0.719 | 0.825 | 70 |
| 30 | 70 | 0.852 | 0.837 | 0.841 | 0.830 | 0.809 | 0.882 | 0.735 | 0.837 | 71 |
| 50 | 50 | 0.859 | 0.845 | 0.849 | 0.838 | 0.817 | 0.889 | 0.746 | 0.845 | 72 |
| 70 | 30 | 0.867 | 0.856 | 0.859 | 0.848 | 0.829 | 0.897 | 0.761 | 0.856 | 73 |
| 100 | 0 | 0.901 | 0.853 | 0.860 | 0.895 | 0.838 | 0.887 | 0.789 | 0.853 | 74 |
