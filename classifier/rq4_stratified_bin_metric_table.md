# RQ4 Stratified Bin Metric Table

Real traces are randomly sampled first, then synthetic traces are sampled to match the real span-bin proportions. Lower JSD, Wasserstein distance, and MAE values indicate better representativeness within each stratum.

## SN

Sampled synthetic bin counts: 1-5=162, 6-10=320, 11-20=265, >20=253.
Synthetic bin shortfalls filled with replacement/fallback: 1-5=0, 6-10=0, 11-20=0, >20=0.

| Metric | 1-5 | 6-10 | 11-20 | >20 |
|---|---:|---:|---:|---:|
| Service Dist. (JSD) | 0.025 | 0.026 | 0.015 | 0.081 |
| Operation Dist. (JSD) | 0.117 | 0.044 | 0.068 | 0.221 |
| Service-Operation Dist. (JSD) | 0.117 | 0.044 | 0.068 | 0.221 |
| Duration Dist. (Wass. log1p) | 0.234 | 0.125 | 0.100 | 0.359 |
| Duration Mean (MAE log1p) | 0.129 | 0.040 | 0.048 | 0.078 |
| Edge Count (MAE) | 7.204 | 9.922 | 4.460 | 96.743 |
| Graph Depth (MAE) | 1.963 | 2.672 | 3.891 | 6.300 |
| Branching Factor (MAE) | 1.458 | 0.732 | 0.759 | 5.730 |

## TT

Sampled synthetic bin counts: 1-5=600, 6-10=190, 11-20=3, >20=207.
Synthetic bin shortfalls filled with replacement/fallback: 1-5=0, 6-10=0, 11-20=0, >20=0.

| Metric | 1-5 | 6-10 | 11-20 | >20 |
|---|---:|---:|---:|---:|
| Service Dist. (JSD) | 0.184 | 0.041 | 0.484 | 0.101 |
| Operation Dist. (JSD) | 0.191 | 0.083 | 0.376 | 0.092 |
| Service-Operation Dist. (JSD) | 0.191 | 0.087 | 1.000 | 1.000 |
| Duration Dist. (Wass. log1p) | 0.380 | 0.346 | 1.183 | 0.147 |
| Duration Mean (MAE log1p) | 0.354 | 0.337 | 0.701 | 0.018 |
| Edge Count (MAE) | 0.320 | 3.295 | 14.333 | 45.961 |
| Graph Depth (MAE) | 0.320 | 1.695 | 3.667 | 3.952 |
| Branching Factor (MAE) | 0.160 | 0.893 | 1.127 | 3.408 |
