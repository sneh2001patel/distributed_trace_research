# RQ4 Stratified Representativeness

Synthetic traces are sampled to match the real trace-size bin proportions. Span bins are small `1-5`, medium `6-10`, large `11-20`, and very large `>20` spans. Lower distance and MAE values indicate closer distributional similarity.

| System | Stratum | Real | Synthetic | Svc JSD | Op JSD | Svc-Op JSD | Dur WDist (log1p) | Dur MAE (log1p) | Edge MAE | Depth MAE | Span MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SN | all_stratified | 25428 | 25428 | 0.066 | 0.180 | 0.180 | 0.274 | 0.072 | 18.720 | 3.744 | 0.000 |
| SN | small | 3816 | 3816 | 0.013 | 0.022 | 0.022 | 0.132 | 0.056 | 6.938 | 1.797 | 0.000 |
| SN | medium | 8314 | 8314 | 0.012 | 0.021 | 0.021 | 0.115 | 0.010 | 9.433 | 2.654 | 0.000 |
| SN | large | 7125 | 7125 | 0.013 | 0.016 | 0.016 | 0.083 | 0.049 | 4.376 | 3.808 | 0.000 |
| SN | very_large | 6173 | 6173 | 0.077 | 0.216 | 0.216 | 0.352 | 0.096 | 99.156 | 6.341 | 0.000 |
| TT | all_stratified | 10942 | 10942 | 0.104 | 0.123 | 0.125 | 0.119 | 0.109 | 9.851 | 1.348 | 5.591 |
| TT | small | 6660 | 6660 | 0.182 | 0.194 | 0.195 | 0.428 | 0.407 | 0.296 | 0.296 | 0.000 |
| TT | medium | 1988 | 1988 | 0.040 | 0.081 | 0.083 | 0.192 | 0.179 | 3.756 | 1.890 | 0.000 |
| TT | large | 16 | 16 | 0.108 | 0.137 | 0.145 | 1.023 | 0.358 | 14.500 | 3.750 | 0.000 |
| TT | very_large | 2278 | 2278 | 0.096 | 0.082 | 0.104 | 0.139 | 0.112 | 43.075 | 3.930 | 26.855 |
