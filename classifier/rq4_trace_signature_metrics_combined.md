# RQ4 Trace Signature Similarity - Combined

Combined SN/TT normal and abnormal trace-signature results. Trace signatures use `(service, operation)` node labels and directed service-operation edge labels.

| System | Class | Real | Synthetic | NN Dist | NN Sim | Dup Rate | Near Dup | Node Cov | Edge Cov | Svc-Op Cov | Unique Rate | Exact Match | Novelty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TT | normal | 1000 | 1000 | 0.400 | 0.600 | 0.655 | 0.655 | 1.000 | 0.647 | 0.854 | 0.378 | 0.462 | 0.538 |
| TT | abnormal | 1000 | 1000 | 0.481 | 0.519 | 0.652 | 0.652 | 0.958 | 0.647 | 0.829 | 0.405 | 0.359 | 0.641 |
| SN | normal | 1000 | 1000 | 0.868 | 0.132 | 0.000 | 0.000 | 1.000 | 0.924 | 0.960 | 1.000 | 0.000 | 1.000 |
| SN | abnormal | 1000 | 1000 | 0.878 | 0.122 | 0.000 | 0.000 | 1.000 | 0.924 | 0.960 | 1.000 | 0.000 | 1.000 |
