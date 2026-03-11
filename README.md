# Learned Optimal Transport Clustering (LOTC)

Official implementation of "Learned Optimal Transport Clustering: A Differentiable Wasserstein Framework for Prototype Learning".

LOTC is a differentiable framework for prototype-based clustering anchored in the principle of atomic measure learning. By minimizing the entropic Sinkhorn distance between an empirical data distribution and a learned discrete target measure, we enable the joint optimization of deep embeddings and cluster prototypes with formal mass conservation.

## Features
- **Theoretical Grounding**: Joint stability guarantees and explicit bias-variance tradeoff bounds.
- **Statistical Rigor**: Multi-seed benchmarks evaluated with Wilcoxon signed-rank tests for significance mapping.
- **Robustness**: Performance maintained under extreme imbalance (up to 1:500) through mass adaptation.
- **Multi-Backbone Support**: Native support for ResNet-18, ResNet-50, and DINO (ViT) extractors.
- **Multi-Modality**: Evaluated across image (CIFAR, STL-10, Tiny-ImageNet) and text (20Newsgroups) domains.

## Repository Structure
- `src/`: Core implementation of LOTC Sinkhorn solvers and clustering objectives.
- `experiments/`: Main benchmark scripts including `run_full_benchmark.py`, `baselines.py`, and `encoders.py`.
- `scripts/`: Assorted reproducibility scripts and visual diagnostics.

## Reproducibility
### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running Full Benchmarks
To reproduce the full multi-backbone LOTC evaluation against SOTA baselines (DEC, SCAN, P²OT):
```bash
python experiments/run_full_benchmark.py --dataset cifar10 --backbone resnet18 --seeds 10
python experiments/run_full_benchmark.py --dataset stl10 --backbone dino --seeds 5
```

### Statistical Analysis
To compute Wilcoxon signed-rank matrices and export LaTeX tables:
```bash
python experiments/statistical_analysis.py --dataset cifar10
```

### Stress Tests
To evaluate robustness against imbalance:
```bash
python scripts/run_stress_tests.py
```

## Citation
If you find this research useful, please cite:
```bibtex
@article{lotc2025,
  title={Learned Optimal Transport Clustering: A Differentiable Wasserstein Framework for Prototype Learning},
  author={...},
  journal={...},
  year={2025}
}
```
