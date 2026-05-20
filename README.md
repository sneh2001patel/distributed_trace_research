# Hierarchical Generation of Synthetic Distributed Traces for Downstream Learning

Authors: Sneh Patel, Yuvraj Sehgal, Mahsa Panahandeh, Naser Ezzati-Jivan, Francois Tetreault

## Overview

Modern microservice systems generate distributed traces that are useful for diagnosis, monitoring, and learning-based analysis. Real traces can be sensitive, expensive to collect, and limited in coverage, so this project studies synthetic trace generation for downstream anomaly detection.

The current experiment compares three synthetic-data generators:

- Hierarchical VAE: a graph-level and node-level variational model that learns trace structure, span attributes, and dependencies.
- Flat VAE: a one-latent graph VAE baseline that generates an entire trace from a single latent representation.
- Empirical/random sampling: a non-learning baseline that samples graph sizes, edge counts, service IDs, operation IDs, and durations from empirical real-data ranges/distributions.

The downstream task is normal-vs-abnormal anomaly detection for the SocialNetwork (SN) and TrainTicket (TT) trace datasets. Classifiers are trained with different real/synthetic mixtures and evaluated on held-out real traces.

Pretrained VAE weights are already included under `./weights/`, and generated synthetic datasets are already present under `./datasets/anomaly/` and `./datasets/baselines/`. If you do not need to retrain or regenerate the VAEs, skip directly to the classifier commands.

## Real Datasets

The real distributed trace data comes from the public Zenodo dataset:

- https://zenodo.org/records/7615394

Raw traces are converted into graph-structured datasets, where each trace is a graph and each node stores span-level attributes such as service, operation, and duration.

Build the base graph datasets with:

```bash
python build_trace_graphs.py
python build_anomaly_trace_graphs.py
```

The anomaly detection scripts expect datasets under:

- `./datasets/anomaly/SN/SN_normal.pt`
- `./datasets/anomaly/SN/SN_abnormal.pt`
- `./datasets/anomaly/TT/TT_normal.pt`
- `./datasets/anomaly/TT/TT_abnormal.pt`

## Hierarchical VAE

The hierarchical VAE is the main generator. It separates trace-level structure from node-level span behavior, allowing the decoder to generate graph topology and node attributes for normal and abnormal traces.

Train the hierarchical VAEs:

```bash
python train_sn_normal_data.py
python train_sn_abnormal_data.py
python train_tt_normal_data.py
python train_tt_abnormal_data.py
```

These commands save weights to:

- `./weights/sn_normal_vae_weights.pt`
- `./weights/sn_abnormal_vae_weights.pt`
- `./weights/tt_normal_vae_weights.pt`
- `./weights/tt_abnormal_vae_weights.pt`

Generate hierarchical VAE synthetic datasets:

```bash
python generate_sn_synthetic_data.py --mode both
python generate_tt_synthetic_data.py --mode both
```

Outputs are written to:

- `./datasets/anomaly/SN/SN_normal_synthetic.pt`
- `./datasets/anomaly/SN/SN_abnormal_synthetic.pt`
- `./datasets/anomaly/TT/TT_normal_synthetic.pt`
- `./datasets/anomaly/TT/TT_abnormal_synthetic.pt`

## Flat VAE Baseline

The flat VAE is a simpler one-latent baseline. It does not explicitly separate graph-level and node-level latent variables, so it provides a comparison point for the hierarchical design.

Train flat VAEs:

```bash
python train_flat_vae.py --data-path ./datasets/anomaly/SN/SN_normal.pt --out-weights ./weights/sn_normal_flat_vae_weights.pt
python train_flat_vae.py --data-path ./datasets/anomaly/SN/SN_abnormal.pt --out-weights ./weights/sn_abnormal_flat_vae_weights.pt
python train_flat_vae.py --data-path ./datasets/anomaly/TT/TT_normal.pt --out-weights ./weights/tt_normal_flat_vae_weights.pt
python train_flat_vae.py --data-path ./datasets/anomaly/TT/TT_abnormal.pt --out-weights ./weights/tt_abnormal_flat_vae_weights.pt
```

Generate flat VAE synthetic datasets:

```bash
python generate_flat_synthetic_data.py --system both --mode both
```

Outputs are written to `./datasets/baselines/*_flat_vae_synthetic.pt`.

## Empirical Random Sampling Baseline

The empirical/random sampling baseline is non-learning. It fits simple empirical properties from the real graph datasets, then samples synthetic graphs from those properties.

Generate all SN and TT normal/abnormal random-sampling baselines:

```bash
python random_sampling_baselines.py --system both --mode both --seed 42
```

Outputs are written to `./datasets/baselines/*_random_sampling.pt`.

There is also a configurable empirical baseline script for one dataset at a time:

```bash
python empirical_baselines.py \
  --real-path ./datasets/anomaly/SN/SN_normal.pt \
  --out-path ./datasets/baselines/SN_normal_empirical_random_tree.pt \
  --y-label 0 \
  --structure random_tree
```

## Downstream Anomaly Detection

The downstream classifier predicts normal vs. abnormal traces and evaluates on held-out real data. By default, the SN and TT classifier scripts use the hierarchical VAE synthetic datasets.

Run a hierarchical VAE classifier test:

```bash
python classifier/train_sn_anomaly.py --real 0 --epochs 50
python classifier/train_tt_anomaly.py --real 0 --epochs 120
```

`--real 0` means the classifier trains on synthetic traces only and tests on real held-out traces. To train with a real/synthetic mixture, pass a real-data percentage:

```bash
python classifier/train_sn_anomaly.py --real 10 --epochs 50
python classifier/train_tt_anomaly.py --real 10 --epochs 120
```

To test a specific generator, pass the synthetic normal and abnormal paths:

```bash
python classifier/train_sn_anomaly.py \
  --generator-name flat_vae \
  --syn-normal-path ./datasets/baselines/SN_normal_flat_vae_synthetic.pt \
  --syn-abnormal-path ./datasets/baselines/SN_abnormal_flat_vae_synthetic.pt \
  --real 0

python classifier/train_tt_anomaly.py \
  --generator-name random_sampling \
  --syn-normal-path ./datasets/baselines/TT_normal_random_sampling.pt \
  --syn-abnormal-path ./datasets/baselines/TT_abnormal_random_sampling.pt \
  --real 0
```

Classifier weights and result CSVs are saved under `./classifier/` unless another path is provided.

## Full Baseline Comparison

Run the ordered comparison across hierarchical VAE, flat VAE, and random sampling:

```bash
python run_baseline_comparison.py \
  --systems SN TT \
  --generators random_sampling flat_vae hierarchical_vae \
  --real-values 0 10 30 50 70 100
```

This writes consolidated metrics to `./classifier/baseline_comparison_ordered_results.csv` and summary tables to `./classifier/baseline_comparison_ordered_tables.md`.

To include the empirical baseline as well:

```bash
python run_baseline_comparison.py \
  --systems SN TT \
  --generators random_sampling empirical flat_vae hierarchical_vae \
  --real-values 0 10 30 50 70 100
```

## Structural Fidelity Evaluation

Structural fidelity scripts compare generated and real graph properties, including node counts, edge counts, density, and related graph statistics.

Run structural fidelity evaluation:

```bash
python structural_fidelity.py --systems SN TT --generators random_sampling flat_vae hierarchical_vae
```

Include the empirical baseline when those datasets have been generated:

```bash
python structural_fidelity.py --systems SN TT --generators random_sampling empirical flat_vae hierarchical_vae
```

The active downstream experiment is anomaly detection with hierarchical VAE, flat VAE, and empirical/random sampling synthetic data.
