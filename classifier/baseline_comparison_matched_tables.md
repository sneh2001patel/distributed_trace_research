# Baseline Comparison Results

Metrics are weighted precision, weighted recall, and weighted F1 on held-out real traces. Normal F1 and Abnormal F1 are included to show class-level behavior.

## TrainTicket

| Generator | Real % | Synthetic % | Precision | Recall | F1 | Normal F1 | Abnormal F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| empirical | 0 | 100 | 0.629 | 0.576 | 0.571 | 0.552 | 0.598 |
| empirical | 10 | 90 | 0.777 | 0.747 | 0.748 | 0.756 | 0.737 |
| empirical | 30 | 70 | 0.782 | 0.745 | 0.746 | 0.750 | 0.741 |
| empirical | 50 | 50 | 0.772 | 0.743 | 0.744 | 0.753 | 0.733 |
| empirical | 70 | 30 | 0.835 | 0.747 | 0.742 | 0.727 | 0.764 |
| empirical | 100 | 0 | 0.841 | 0.742 | 0.736 | 0.718 | 0.763 |
| flat_vae | 0 | 100 | 0.678 | 0.647 | 0.648 | 0.656 | 0.638 |
| flat_vae | 10 | 90 | 0.740 | 0.732 | 0.733 | 0.758 | 0.698 |
| flat_vae | 30 | 70 | 0.799 | 0.741 | 0.740 | 0.735 | 0.748 |
| flat_vae | 50 | 50 | 0.769 | 0.742 | 0.744 | 0.753 | 0.730 |
| flat_vae | 70 | 30 | 0.838 | 0.741 | 0.735 | 0.717 | 0.761 |
| flat_vae | 100 | 0 | 0.841 | 0.742 | 0.736 | 0.718 | 0.763 |
| hierarchical_vae | 0 | 100 | 0.544 | 0.462 | 0.411 | 0.303 | 0.562 |
| hierarchical_vae | 10 | 90 | 0.764 | 0.741 | 0.742 | 0.754 | 0.726 |
| hierarchical_vae | 30 | 70 | 0.830 | 0.738 | 0.732 | 0.715 | 0.757 |
| hierarchical_vae | 50 | 50 | 0.840 | 0.741 | 0.736 | 0.717 | 0.762 |
| hierarchical_vae | 70 | 30 | 0.841 | 0.742 | 0.736 | 0.718 | 0.763 |
| hierarchical_vae | 100 | 0 | 0.841 | 0.742 | 0.736 | 0.718 | 0.763 |

## SocialNetwork

| Generator | Real % | Synthetic % | Precision | Recall | F1 | Normal F1 | Abnormal F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| empirical | 0 | 100 | 0.606 | 0.329 | 0.259 | 0.195 | 0.424 |
| empirical | 10 | 90 | 0.766 | 0.779 | 0.767 | 0.855 | 0.537 |
| empirical | 30 | 70 | 0.819 | 0.815 | 0.817 | 0.870 | 0.677 |
| empirical | 50 | 50 | 0.880 | 0.838 | 0.845 | 0.877 | 0.763 |
| empirical | 70 | 30 | 0.865 | 0.837 | 0.843 | 0.879 | 0.750 |
| empirical | 100 | 0 | 0.899 | 0.852 | 0.859 | 0.887 | 0.787 |
| flat_vae | 0 | 100 | 0.582 | 0.659 | 0.607 | 0.787 | 0.139 |
| flat_vae | 10 | 90 | 0.856 | 0.820 | 0.827 | 0.864 | 0.730 |
| flat_vae | 30 | 70 | 0.835 | 0.823 | 0.827 | 0.873 | 0.706 |
| flat_vae | 50 | 50 | 0.882 | 0.843 | 0.850 | 0.882 | 0.768 |
| flat_vae | 70 | 30 | 0.865 | 0.846 | 0.851 | 0.888 | 0.754 |
| flat_vae | 100 | 0 | 0.902 | 0.853 | 0.860 | 0.887 | 0.789 |
| hierarchical_vae | 0 | 100 | 0.599 | 0.673 | 0.620 | 0.797 | 0.160 |
| hierarchical_vae | 10 | 90 | 0.856 | 0.819 | 0.826 | 0.864 | 0.729 |
| hierarchical_vae | 30 | 70 | 0.873 | 0.843 | 0.849 | 0.883 | 0.761 |
| hierarchical_vae | 50 | 50 | 0.875 | 0.843 | 0.849 | 0.883 | 0.762 |
| hierarchical_vae | 70 | 30 | 0.872 | 0.848 | 0.853 | 0.888 | 0.763 |
| hierarchical_vae | 100 | 0 | 0.900 | 0.852 | 0.859 | 0.887 | 0.787 |
