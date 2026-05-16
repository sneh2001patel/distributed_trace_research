# RQ4 Trace Signature Similarity

Trace signature = set of `(service, operation)` node labels plus set of directed edge labels `(parent_service, parent_operation) -> (child_service, child_operation)`. Nearest-neighbor distance is Jaccard distance over the combined node/edge signature items.

| System | Preset | Real | Synthetic | NN Dist | NN Sim | Dup Rate | Near Dup | Node Cov | Edge Cov | Unique Rate | Novelty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SN | fixed_size_precise | 1244 | 1244 | 0.940 | 0.060 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TT | fixed_size_precise | 1244 | 1244 | 0.886 | 0.114 | 0.466 | 0.466 | 1.000 | 1.000 | 0.581 | 0.982 |
