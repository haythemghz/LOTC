# Learned Optimal Transport Clustering (LOTC)

Official implementation of "Learned Optimal Transport Clustering: A Differentiable Wasserstein Framework for Prototype Learning".

LOTC is a differentiable framework for prototype-based clustering anchored in the principle of atomic measure learning. By minimizing the entropic Sinkhorn distance between an empirical data distribution and a learned discrete target measure, we enable the joint optimization of deep embeddings and cluster prototypes with formal mass conservation.

## Features
- **Theoretical Grounding**: Joint stability guarantees and bias-variance analysis.
- **Statistical Rigor**: 10-seed benchmarks with paired t-tests and Cohen's d.
- **Robustness**: Performance maintained under extreme imbalance (up to 1:500) and feature noise.
- **Scalability**: $O(BK)$ memory complexity and efficient unrolled Sinkhorn gradients.

## Repository Structure
- `src/`: Core implementation of LOTC and OT solvers.
- `scripts/`: Reproducibility scripts for benchmarks, scaling studies, and stress tests.
- `experiments/`: Configuration files and summary results.

## Reproducibility
### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running Benchmarks
To reproduce the rigorous 10-seed benchmarks:
```bash
python scripts/run_rigorous_eval.py --dataset stl10
```

### Stress Tests
To evaluate robustness against imbalance and noise:
```bash
python scripts/run_stress_tests.py
```

### Scaling Study
To reproduce the runtime/memory scaling plots:
```bash
python scripts/benchmark_grid.py
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
