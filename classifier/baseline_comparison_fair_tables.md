# Fair Baseline Comparison Results

Metrics are weighted precision, weighted recall, and weighted F1 on held-out real traces. The comparison caps eligible synthetic data to the same per-class budget across empirical, flat VAE, and hierarchical VAE runs.

## TrainTicket

| Generator | Real % | Synthetic % | Precision | Recall | F1 | Normal F1 | Abnormal F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| empirical | 0 | 100 | 0.612 | 0.551 | 0.540 | 0.505 | 0.589 |
| empirical | 10 | 90 | 0.804 | 0.722 | 0.717 | 0.700 | 0.740 |
| empirical | 30 | 70 | 0.832 | 0.728 | 0.720 | 0.698 | 0.752 |
| empirical | 50 | 50 | 0.829 | 0.728 | 0.722 | 0.700 | 0.752 |
| empirical | 70 | 30 | 0.830 | 0.726 | 0.719 | 0.696 | 0.751 |
| empirical | 100 | 0 | 0.835 | 0.727 | 0.719 | 0.695 | 0.753 |
| flat_vae | 0 | 100 | 0.697 | 0.688 | 0.665 | 0.770 | 0.517 |
| flat_vae | 10 | 90 | 0.796 | 0.715 | 0.710 | 0.693 | 0.734 |
| flat_vae | 30 | 70 | 0.831 | 0.727 | 0.720 | 0.697 | 0.752 |
| flat_vae | 50 | 50 | 0.833 | 0.727 | 0.719 | 0.696 | 0.752 |
| flat_vae | 70 | 30 | 0.835 | 0.727 | 0.719 | 0.695 | 0.753 |
| flat_vae | 100 | 0 | 0.836 | 0.728 | 0.720 | 0.696 | 0.753 |
| hierarchical_vae | 0 | 100 | 0.512 | 0.494 | 0.497 | 0.523 | 0.461 |
| hierarchical_vae | 10 | 90 | 0.790 | 0.715 | 0.711 | 0.697 | 0.732 |
| hierarchical_vae | 30 | 70 | 0.833 | 0.728 | 0.721 | 0.699 | 0.753 |
| hierarchical_vae | 50 | 50 | 0.830 | 0.727 | 0.720 | 0.697 | 0.751 |
| hierarchical_vae | 70 | 30 | 0.832 | 0.726 | 0.719 | 0.695 | 0.751 |
| hierarchical_vae | 100 | 0 | 0.836 | 0.728 | 0.720 | 0.696 | 0.753 |

## SocialNetwork

| Generator | Real % | Synthetic % | Precision | Recall | F1 | Normal F1 | Abnormal F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| empirical | 0 | 100 | 0.606 | 0.329 | 0.259 | 0.195 | 0.424 |
| empirical | 10 | 90 | 0.822 | 0.815 | 0.818 | 0.869 | 0.684 |
| empirical | 30 | 70 | 0.828 | 0.822 | 0.825 | 0.875 | 0.694 |
| empirical | 50 | 50 | 0.808 | 0.811 | 0.810 | 0.871 | 0.651 |
| empirical | 70 | 30 | 0.881 | 0.840 | 0.847 | 0.879 | 0.764 |
| empirical | 100 | 0 | 0.899 | 0.852 | 0.859 | 0.887 | 0.786 |
| flat_vae | 0 | 100 | 0.561 | 0.554 | 0.557 | 0.688 | 0.215 |
| flat_vae | 10 | 90 | 0.794 | 0.788 | 0.791 | 0.851 | 0.634 |
| flat_vae | 30 | 70 | 0.838 | 0.820 | 0.825 | 0.869 | 0.710 |
| flat_vae | 50 | 50 | 0.883 | 0.839 | 0.846 | 0.877 | 0.765 |
| flat_vae | 70 | 30 | 0.864 | 0.843 | 0.848 | 0.885 | 0.751 |
| flat_vae | 100 | 0 | 0.898 | 0.851 | 0.858 | 0.886 | 0.785 |
| hierarchical_vae | 0 | 100 | 0.599 | 0.672 | 0.620 | 0.796 | 0.160 |
| hierarchical_vae | 10 | 90 | 0.789 | 0.785 | 0.787 | 0.849 | 0.624 |
| hierarchical_vae | 30 | 70 | 0.868 | 0.821 | 0.829 | 0.863 | 0.741 |
| hierarchical_vae | 50 | 50 | 0.854 | 0.840 | 0.844 | 0.885 | 0.738 |
| hierarchical_vae | 70 | 30 | 0.855 | 0.841 | 0.845 | 0.886 | 0.740 |
| hierarchical_vae | 100 | 0 | 0.902 | 0.853 | 0.860 | 0.887 | 0.789 |
