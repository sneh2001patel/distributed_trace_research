# RQ4 Trace Signature Similarity

Trace signature = set of `(service, operation)` node labels plus set of directed edge labels `(parent_service, parent_operation) -> (child_service, child_operation)`. Nearest-neighbor distance is Jaccard distance over the combined node/edge signature items.

| System | Preset | Real | Synthetic | NN Dist | NN Sim | Dup Rate | Near Dup | Node Cov | Edge Cov | Unique Rate | Novelty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SN | anomaly_abnormal | 1000 | 1000 | 0.878 | 0.122 | 0.000 | 0.000 | 1.000 | 0.924 | 1.000 | 1.000 |
