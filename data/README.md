# Data Notes

This repository does not bundle large raw benchmark traces or proprietary telemetry.

The experiments use public TrainTicket and SocialNetwork microservice benchmark traces after preprocessing into graph objects. The final result tables and summaries are included under `results/icsme2026_final/`.

To rerun the full pipeline, place the benchmark trace files and generated graph datasets in the paths expected by the scripts in `src/`.
