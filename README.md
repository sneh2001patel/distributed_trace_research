# Types of Encoders [^1]

- GCN (Type of GNN)
- VAE (Diffusion Encoder)
- Trace-BERT (token based)

codex resume 019a5066-3c65-77a1-8024-46600fc61299
[^1]: Just to start off with

## Ways to improve GCN Model:

| Improvement                             | Expected Gains |
| --------------------------------------- | -------------- |
| Add 3rd GCNConv Layer                   | +2-4%          |
| Switch to GAN (attention)               | +3-6%          |
| Use GraphSage                           | +1-3%          |
| Use attention pooling instead of global | +3-5%          |
| Use deeper MLP head                     | +1-2%          |
| Increase embeded_dim to 64-96           | +2-3%          |
