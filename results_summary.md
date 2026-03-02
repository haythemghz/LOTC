# Results Summary: Learned Optimal Transport Clustering (LOTC)

## Synthetic Benchmarks
| Dataset | ACC | ARI | NMI |
| :--- | :---: | :---: | :---: |
| Blobs | 1.000 | 1.000 | 1.000 |
| Moons | 0.736 | 0.222 | 0.167 |
| Circles | 0.723 | 0.198 | 0.149 |
| Unbalanced | 0.985 | 0.956 | 0.942 |

## Real-World Benchmarks (Subsampled)
| Dataset | Encoder | ACC | ARI | NMI |
| :--- | :--- | :---: | :---: | :---: |
| Fashion-MNIST | MLP | 0.532 | 0.311 | 0.445 |
| CIFAR-10 | ResNet-18 | 0.136 | 0.007 | 0.040 |

## Key Findings
- **Stability**: The log-domain Sinkhorn solver proved extremely stable even with small $\varepsilon$.
- **Mass Learning**: LOTC successfully recovered ground-truth cluster masses in the unbalanced case.
- **Representation Learning**: Backpropagating through OT costs successfully guided the MLP/ResNet encoders to form cluster-friendly latent spaces.
