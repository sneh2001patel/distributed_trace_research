# Hierarchical Generation of Synthetic Distributed Traces for Downstream Learning

Authors: Sneh Patel, Yuvraj Seghal, Mahsa Panahandeh, Naser Ezzati-Jivan, Francois Tetreau

## Overview

Modern microservice-based systems generate large volumes of distributed execution traces, which are valuable for monitoring, diagnosis, and learning-based analysis. However, real traces are often sensitive, costly to collect, and limited in diversity.

This project proposes a hierarchical generative framework for synthesizing realistic distributed traces that preserve both:

- Global structure (trace-level and graph-level properties), and

- Local semantics (span- and event-level attributes).

The generated synthetic traces are evaluated not only for statistical similarity to real traces, but also for utility in downstream learning tasks, such as classification and prediction, using train-on-synthetic-test-on-real (TSTR) evaluation

## Real datasets

The real distributed trace data used in this project was obtained from the following public dataset:

- Zenodo Dataset: https://zenodo.org/records/7615394

This dataset contains real-world distributed traces collected from microservice-based benchmarks and serves as the foundation for all training and evaluation in this work.

After downloading the raw traces, they are preprocessed and converted into graph-structured representations suitable for hierarchical generative modeling. The trace-to-graph conversion is performed using the following script `./build_trace_graphs.py`

This preprocessing step transforms each distributed trace into a graph where nodes and edges encode span-level structure and attributes, enabling learning over both global trace structure and local execution semantics.

## Encoder-Decoder Architecture

The hierarchical encoder–decoder models used in this work can be trained using the following scripts:

- `./train_tt_data.py` — training on the TrainTicket (TT) benchmark

- `./train_sn_data.py` — training on the SocialNetwork (SN) benchmark

These scripts train the encoder–decoder architecture to learn both global trace structure and node-level attributes, enabling the decoder to generate synthetic graph-structured traces that resemble real distributed executions.

During training, the learned model parameters are automatically saved to the `weights/` directory. These saved weights are later used during sampling to generate synthetic traces for evaluation and downstream learning tasks.

## Synthetic Datasets Generation

After training the hierarchical encoder–decoder models, synthetic distributed trace datasets are generated using the following script `./build_synthetic_datasets.py`
The datasets are saved in the directory `./datasets/`. Two different types of datasets where generated fixed-size and variable-size. Fixed-size indicated that the dataset was made to mimic the real dataset, where the variable-size dataset had liberty.
The resulting synthetic datasets are saved in the same standardized formats used by the real traces, enabling them to be directly substituted into downstream learning pipelines without modification.

## Downstream Task: Classification

The generated synthetic datasets are subsequently used to evaluate their utility in downstream learning tasks. In particular, we train predictive models exclusively on synthetic data and evaluate them on held-out real data, following a train-on-synthetic-test-on-real (TSTR) evaluation protocol.

These downstream experiments assess whether the synthetic traces capture sufficient behavioral fidelity to support learning tasks such as classification and prediction, and serve as a key measure of synthetic data quality in this work.

The `./classifier/` directory has the model, training method used, and the evaluation done, along with the weights.

## Similarity and Fidelity Evaluation

To assess how closely the generated synthetic datasets resemble the real distributed traces, we perform a comprehensive similarity and fidelity analysis across multiple dimensions.

The evaluation includes:

Statistical similarity:
Distributional comparisons of key attributes (e.g., service identifiers, operation identifiers, durations) between real and synthetic traces using metrics such as Kolmogorov–Smirnov (KS) statistics.

Structural similarity:
Analysis of graph-level properties to compare real and synthetic trace structures, ensuring that generated graphs preserve essential execution patterns and dependencies.

Clustering-based distinguishability:
Unsupervised clustering (e.g., K-means) over engineered feature representations to evaluate whether real and synthetic traces are well mixed or easily separable.

Together, these analyses provide complementary evidence that the synthetic datasets capture both the structural characteristics and behavioral variability of real distributed traces, while avoiding direct memorization. This similarity evaluation complements downstream task performance and provides a principled assessment of synthetic data quality.
