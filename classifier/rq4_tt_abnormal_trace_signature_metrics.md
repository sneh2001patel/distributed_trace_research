# RQ4 Trace Signature Similarity

Trace signature = set of `(service, operation)` node labels plus set of directed edge labels `(parent_service, parent_operation) -> (child_service, child_operation)`. Nearest-neighbor distance is Jaccard distance over the combined node/edge signature items.

| System | Preset | Real | Synthetic | NN Dist | NN Sim | Dup Rate | Near Dup | Node Cov | Edge Cov | Unique Rate | Novelty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TT | anomaly_abnormal | 1000 | 1000 | 0.481 | 0.519 | 0.652 | 0.652 | 0.958 | 0.647 | 0.405 | 0.641 |
