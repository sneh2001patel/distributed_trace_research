# Hierarchical-Fixed Baseline Comparison Results

Metrics are weighted precision, weighted recall, and weighted F1 on held-out real traces. Normal F1 and Abnormal F1 are included to show class-level behavior.

## TrainTicket

| Generator | Real % | Synthetic % | Precision | Recall | F1 | Normal F1 | Abnormal F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| empirical | 0 | 100 | 0.612 | 0.551 | 0.540 | 0.505 | 0.589 |
| empirical | 10 | 90 | 0.809 | 0.718 | 0.712 | 0.691 | 0.740 |
| empirical | 30 | 70 | 0.832 | 0.728 | 0.720 | 0.698 | 0.752 |
| empirical | 50 | 50 | 0.829 | 0.728 | 0.721 | 0.699 | 0.751 |
| empirical | 70 | 30 | 0.836 | 0.729 | 0.722 | 0.699 | 0.754 |
| empirical | 100 | 0 | 0.836 | 0.728 | 0.720 | 0.696 | 0.753 |
| flat_vae | 0 | 100 | 0.697 | 0.688 | 0.665 | 0.770 | 0.517 |
| flat_vae | 10 | 90 | 0.786 | 0.712 | 0.707 | 0.692 | 0.728 |
| flat_vae | 30 | 70 | 0.823 | 0.725 | 0.719 | 0.698 | 0.748 |
| flat_vae | 50 | 50 | 0.833 | 0.728 | 0.720 | 0.697 | 0.752 |
| flat_vae | 70 | 30 | 0.835 | 0.727 | 0.719 | 0.695 | 0.753 |
| flat_vae | 100 | 0 | 0.836 | 0.728 | 0.720 | 0.696 | 0.753 |
| hierarchical_vae | 0 | 100 | 0.665 | 0.666 | 0.665 | 0.718 | 0.591 |
| hierarchical_vae | 10 | 90 | 0.810 | 0.719 | 0.713 | 0.694 | 0.741 |
| hierarchical_vae | 30 | 70 | 0.836 | 0.728 | 0.720 | 0.696 | 0.753 |
| hierarchical_vae | 50 | 50 | 0.832 | 0.725 | 0.718 | 0.694 | 0.751 |
| hierarchical_vae | 70 | 30 | 0.832 | 0.725 | 0.718 | 0.694 | 0.751 |
| hierarchical_vae | 100 | 0 | 0.836 | 0.728 | 0.720 | 0.696 | 0.753 |

## SocialNetwork

| Generator | Real % | Synthetic % | Precision | Recall | F1 | Normal F1 | Abnormal F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| empirical | 0 | 100 | 0.606 | 0.329 | 0.259 | 0.195 | 0.424 |
| empirical | 10 | 90 | 0.825 | 0.813 | 0.817 | 0.866 | 0.689 |
| empirical | 30 | 70 | 0.828 | 0.822 | 0.824 | 0.875 | 0.693 |
| empirical | 50 | 50 | 0.816 | 0.817 | 0.817 | 0.874 | 0.667 |
| empirical | 70 | 30 | 0.889 | 0.837 | 0.845 | 0.874 | 0.768 |
| empirical | 100 | 0 | 0.899 | 0.852 | 0.859 | 0.887 | 0.786 |
| flat_vae | 0 | 100 | 0.561 | 0.554 | 0.557 | 0.688 | 0.215 |
| flat_vae | 10 | 90 | 0.799 | 0.790 | 0.793 | 0.851 | 0.644 |
| flat_vae | 30 | 70 | 0.844 | 0.820 | 0.826 | 0.868 | 0.717 |
| flat_vae | 50 | 50 | 0.877 | 0.841 | 0.847 | 0.880 | 0.762 |
| flat_vae | 70 | 30 | 0.872 | 0.842 | 0.848 | 0.882 | 0.759 |
| flat_vae | 100 | 0 | 0.902 | 0.853 | 0.860 | 0.887 | 0.789 |
| hierarchical_vae | 0 | 100 | 0.597 | 0.422 | 0.434 | 0.451 | 0.390 |
| hierarchical_vae | 10 | 90 | 0.798 | 0.796 | 0.797 | 0.858 | 0.636 |
| hierarchical_vae | 30 | 70 | 0.809 | 0.817 | 0.810 | 0.878 | 0.630 |
| hierarchical_vae | 50 | 50 | 0.861 | 0.849 | 0.853 | 0.892 | 0.751 |
| hierarchical_vae | 70 | 30 | 0.880 | 0.843 | 0.849 | 0.882 | 0.766 |
| hierarchical_vae | 100 | 0 | 0.898 | 0.851 | 0.858 | 0.886 | 0.786 |
