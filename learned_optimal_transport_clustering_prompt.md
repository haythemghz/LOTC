# Learned Optimal Transport Clustering — Full prompt for an AI engineer ("antigravity")

> Goal: implement from A→Z a production‑grade, reproducible research project that (1) proposes a *Learned Optimal Transport Clustering* algorithm, (2) provides theoretical justification and analysis, (3) produces extensive experiments and ablations across synthetic and real benchmarks, and (4) writes a final paper suitable for submission to a top ML journal (e.g., JMLR / TPAMI) or conference (NeurIPS / ICML). Deliverables must include code, notebooks, Docker/conda environment, datasets download & preprocessing scripts, all experiment logs, figures, and a compiled LaTeX paper + supplementary material.

---

## High-level summary of the method

Formulate clustering as the learning of an atomic probability measure (a set of weighted prototypes) which minimizes an entropic optimal transport distance to the empirical data distribution. The algorithm jointly optimizes:

- prototype locations \(\{c_j\}_{j=1}^K\) in a feature space (either original feature space or learned embedding),
- prototype masses \(\{\alpha_j\}_{j=1}^K\) (nonnegative, summing to 1),
- (optional) a parametric embedding function \(\phi_\theta(x)\) (neural encoder) mapping raw inputs to a representation space where transport cost is measured.

The central objective (entropic OT) is:

\[
\min_{\theta, c, \alpha}\; \operatorname{OT}_\varepsilon\big(\hat\mu_n, \nu_{c,\alpha}\big) + \lambda_R R(c,\alpha,\theta),
\]

where

- \(\hat\mu_n = \frac{1}{n} \sum_{i=1}^n \delta_{\phi_\theta(x_i)}\) is the empirical distribution in representation space;  
- \(\nu_{c,\alpha} = \sum_{j=1}^K \alpha_j \delta_{c_j}\) is the atomic (prototype) measure;  
- \(\operatorname{OT}_\varepsilon\) denotes entropic‑regularized optimal transport (Sinkhorn), with cost matrix \(C_{ij} = d\big(\phi_\theta(x_i), c_j\big)^2\) (or any differentiable cost);  
- \(R(\cdot)\) is a regularizer (mass prior, prototype dispersion, Laplacian on prototypes, or prototype sparsity);  
- \(\lambda_R\) and \(\varepsilon\) are hyperparameters.

Key properties to implement and exploit:

- Differentiable Sinkhorn to backpropagate OT loss w.r.t. prototype locations and embedding parameters.  
- Possibility to *learn masses* \(\alpha_j\) (soft cluster sizes) or fix them (uniform).  
- Optional geometric regularization to encourage prototypes to align with data manifold (e.g., graph Laplacian on prototypes or TV penalty).  
- Optionally use **Wasserstein‑2** geometry or squared Euclidean cost; support learned ground cost via a trainable Mahalanobis matrix or embedding.


---

## Concrete mathematical formulation and algorithmic building blocks

### Sinkhorn OT objective (discrete, entropic)
Given cost matrix \(C\in\mathbb{R}^{n\times K}\), entropic OT with marginals \(u=\frac{1}{n}1_n\) and \(v=\alpha\) solves

\[
P^* = \arg\min_{P\in\mathbb{R}_+^{n\times K}} \langle P, C \rangle - \varepsilon H(P)\quad\text{s.t.}\quad P\mathbf{1}_K = u, \; P^T\mathbf{1}_n = v,
\]

with \(H(P)= -\sum_{ij} P_{ij}\log P_{ij}\). Use the standard Sinkhorn iterations (matrix scaling) to compute \(P^*\) efficiently and stably in the log domain when \(\varepsilon\) is small.

The transport cost value is \(\mathrm{OT}_\varepsilon(u,v;C) = \langle P^*, C\rangle\).

### Differentiability
Implement the differentiable Sinkhorn by either

1. differentiating through a fixed number of Sinkhorn iterations via autograd (practical), or
2. using implicit differentiation / envelope theorem to get gradients through the fixed‑point (more advanced).  

For research reproducibility, implement option (1) first (unrolled Sinkhorn), include option (2) in supplementary if needed.

### Optimization variables and constraints
- Prototypes \(c_j\): unconstrained vectors in \(\mathbb{R}^d\).  
- Masses \(\alpha_j\): parametrize as softmax of unconstrained logits to ensure nonnegativity and sum to 1.  
- Embedding \(\phi_\theta\): optional neural network (small MLP for tabular, CNN backbone for images). Use batch normalization and optional projection head.

### Loss and regularizers (detailed)
- OT loss: \(L_{OT} = \mathrm{OT}_\varepsilon(u,v;C)\).  
- Mass entropy regularizer: \(R_{mass}=\eta\sum_j \alpha_j\log\alpha_j\) (prefer balanced clusters when desired).  
- Prototype dispersion: \(R_{disp} = \rho\sum_j \|c_j\|^2\) or promote spread by penalizing prototype collisions.  
- Graph Laplacian regularizer: construct a kNN graph on data or prototypes and add \(c^T L c\) to encourage manifold alignment.  
- (Optional) Contrastive/cluster-friendly representation loss: add small auxiliary loss (e.g., self‑supervised BYOL/SimCLR style or cluster assignment entropy objective) to avoid collapse when using learned embedding.

Total loss (minimize):

\[
L = L_{OT} + \lambda_{mass} R_{mass} + \lambda_{disp} R_{disp} + \lambda_{lap} R_{lap} + \lambda_{aux} L_{aux}.
\]

### Algorithm (high level)
1. Initialize prototypes \(c_j\) (KMeans on initial embeddings, or random sample of points), initialize mass logits, initialize encoder \(\theta\) (if used).  
2. For each training epoch:  
   a. Compute embeddings \(Z = \phi_\theta(X)\) (or use raw X).  
   b. Compute cost matrix \(C_{ij}=\|z_i - c_j\|^2\) or learned cost.  
   c. Run Sinkhorn for fixed T iterations to obtain transport plan \(P\) and OT value.  
   d. Compute total loss \(L\) and backpropagate to update \(\theta, c, \text{mass logits}\) via Adam/SGD.  
   e. Optionally, reproject prototypes (e.g., moving average on centroids) and renormalize masses.  
3. Stop after convergence / fixed epochs.  
4. For final hard assignments: cluster label for point i is \(\arg\max_j P_{ij}\) or assign to prototype with highest assigned mass.

Provide both soft and hard assignments in outputs.

---

## Implementation requirements (code & structure)

- Language: Python 3.9+. Primary frameworks: PyTorch (>=1.10). Use JAX as an alternative branch if desired.  
- Repository structure (recommended):

```
learned-ot-clustering/
├── README.md
├── environment.yml          # conda env with pinned versions
├── Dockerfile               # reproducible runtime
├── setup.py or pyproject.toml
├── src/
│   ├── data/                # dataset download & preprocessing scripts
│   ├── models/              # encoder, projection head, prototype module
│   ├── ot/                  # sinkhorn implementation (differentiable)
│   ├── training/            # training loops, trainers, logging
│   ├── eval/                # metrics, baselines wrappers, statistical tests
│   ├── experiments/         # experiment configs (yaml) and runners
│   └── utils/               # helpers
├── notebooks/               # demo notebooks (synthetic, MNIST)
├── experiments/             # saved results, logs, figures
├── paper/                   # LaTeX source + figures + supplementary
└── scripts/                 # run_experiment.sh, evaluate_all.sh
```

- Implement unit tests for Sinkhorn (small handcrafted example), prototype updates, and final assignment computation.
- Logging: use Weights & Biases or TensorBoard for run tracking; save final checkpoints and transport matrices for analysis.

---

## Detailed implementation notes and tips

### Sinkhorn implementation
- Implement numerically stable Sinkhorn in the log domain (stabilized scaling) to handle small \(\varepsilon\).  
- Provide both CPU and GPU implementations; vectorize across batches.  
- Allow `max_iter` and `tolerance` stopping criteria; for autograd unrolling choose a moderate fixed iteration count (e.g., 50).  
- For efficiency, if dataset is large, implement mini‑batch OT via stochastic or sliced OT approximations. Provide both full OT (for medium datasets) and mini‑batch variant for large datasets.

### Handling large n (scalability)
- Provide two modes:
  - **Full OT**: use when n is small to medium (n ≤ 50k) and GPU memory permits; batch‑tiled cost computation.  
  - **Mini‑batch / stochastic OT**: sample minibatches of data and prototypes, use partial Sinkhorn and momentum updates on prototypes; or use Sinkhorn with subsampling and debiasing.  
- Optionally implement *sliced Wasserstein* or *Sinkhorn on clusters of points* strategy to scale up.

### Initialization
- KMeans on initial embeddings is a robust initializer for prototypes.  
- Mass logits initialized uniformly (same value) unless prior knowledge suggests otherwise.

### Hyperparameters to expose
- K (number of clusters), \(\varepsilon\) (Sinkhorn entropic reg), sinkhorn_iter, learning rates (encoder, prototypes, mass logits), regularization weights (\(\lambda_{*}\)), embedding dim, batch size, epochs.

---

## Datasets (download & preprocessing)

**Synthetic** (implement generator scripts):
- 2‑D: blobs, moons, circles, concentric rings, overlapping Gaussian mixtures (vary variances and overlaps).  
- Complex surfaces: Swiss roll with clusters, nested clusters with bridges.

**Benchmark suites and realistic datasets**:
- FCPS / clustering benchmark suite (use multiple instances).  
- UCI tabular datasets: Iris, Wine, Adult, Heart Disease (preprocess as usual).  
- Image: MNIST, Fashion‑MNIST, CIFAR‑10 (subset or use pretrained features), STL10 (if compute allows).  
- Text: 20Newsgroups (TF‑IDF embeddings or pretrained transformer embeddings).  
- Single‑cell RNA‑seq: PBMC 3k (or equivalent public scRNA datasets) — preprocess (normalization, log1p, HVG selection).  
- Time‑series: selected tasks from UCR archive for clustering.  

For high‑dimensional data (images / text / scRNA): use a two‑stage approach: (A) use pretrained encoders (e.g., ResNet18 pretrained on ImageNet, or BERT sentence embeddings) as baseline embeddings; (B) optionally train small domain encoder end‑to‑end for better performance.

**Data splits & reproducibility**
- For each dataset, provide a deterministic preprocessing pipeline and a seedable RNG.  
- For each experimental condition, run with 10 random seeds and store per‑seed outputs.

---

## Baselines to compare against (implementation or wrappers)

Classical:
- KMeans (sklearn)  
- GMM (sklearn)  
- Agglomerative / Ward  
- DBSCAN / HDBSCAN  
- Spectral Clustering  

Deep / modern:
- DEC (Deep Embedding Clustering)  
- DeepCluster  
- SCAN  
- IMSAT / contrastive deep clustering methods  

OT‑related baselines:
- KMeans in learned embedding (with the same encoder)  
- Barycenter clustering heuristics (if available)  

Implement wrappers so that every baseline uses consistent preprocessing, embeddings, and compute‑budgeted hyperparameter tuning.

---

## Evaluation protocol (rigorous, journal‑grade)

### Metrics (labelled datasets)
- Primary: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Adjusted Mutual Information (AMI).  
- Secondary: Homogeneity, Completeness, V‑measure.  
- For downstream usefulness: supervised classification accuracy when using cluster labels as features or as pseudo‑labels for training a classifier.

### Unsupervised / intrinsic metrics (no labels)
- Silhouette Score, Calinski‑Harabasz, Davies‑Bouldin.  
- Stability under subsampling: pairwise ARI between clusterings trained on random 80% subsamples.  

### Robustness evaluations
- Additive noise experiments: Gaussian noise levels, outlier insertion (random noise points), feature corruption (missing values, occlusion for images).  
- Vary sample size n and cluster imbalance.  

### Scalability & resource metrics
- Wall-clock runtime, peak GPU memory, and CPU time per epoch.  
- Complexity vs n, K and embedding dim.

### Experimental protocol details
- For each dataset-method pair: run experiments with 10 different random seeds.  
- For hyperparameters: define a fixed compute budget per dataset (e.g., 50 trials of random search with the same budget for each method).  
- Report mean ± standard deviation across seeds for primary metrics.  
- Statistical testing: use Wilcoxon signed‑rank test for pairwise comparisons across seeds; use Friedman + Nemenyi posthoc for multiple comparisons across many methods/datasets.  
- Present critical difference diagrams where appropriate.

---

## Ablations and analyses to include

1. Entropic regularization \(\varepsilon\) sweep (small → large) and effect on clustering quality & transport sparsity.  
2. Learnable masses vs fixed uniform masses.  
3. Learned embedding vs fixed pretrained embedding (for images/text).  
4. Prototype count K sensitivity.  
5. Mini‑batch OT vs full OT (scalability tradeoffs).  
6. Different cost functions (Euclidean squared, Mahalanobis, cosine).  
7. Regularizers on prototypes (laplacian, dispersion).  
8. Effect of initialization (random vs KMeans init).  

For each ablation, include quantitative metrics, qualitative visualization (transport plans, prototype trajectories during training), and runtime/memory profiles.

---

## Visualization & interpretability

- Plot 2‑D synthetic clusters with prototype locations and transport arrows (heatmap of P).  
- For images: visualize prototypical images corresponding to prototypes (nearest data points in embedding space) and show transport mass distribution.  
- Display confusion matrices vs ground truth, per‑cluster precision/recall.  
- Show convergence plots: OT loss, prototype movement (L2 norm), ARI/NMI vs epoch.  
- Visualize learned masses and their relation to cluster sizes.

---

## Statistical rigor and reproducibility

- Use seeds for numpy, torch, and any other RNGs; save seeds in logs.  
- Provide `environment.yml` and `Dockerfile` with exact package versions.  
- Provide `run_all_experiments.sh` that reproduces the main figures and results tables given adequate compute.  
- Provide automated unit tests and a minimal smoke test dataset (tiny synthetic) that runs in CI (e.g., GitHub Actions).  
- Publish final artifacts: code, preprocessed datasets (if licensing permits), experiment logs, random seeds, model checkpoints, and a reproducibility checklist.

---

## Paper writing stage — required contents

The AI must generate a LaTeX manuscript with the following structure and content:

1. **Title & abstract**: concise, emphasise learned OT as a unifying clustering framework and key contributions (theory + empirical).  
2. **Introduction**: position relative to clustering literature, OT literature, and deep clustering; clearly state research gap and contributions (theorem statements; algorithm; extensive benchmarks; reproducible artifacts).  
3. **Related work**: survey kmeans/GMM/spectral/OT clustering, deep clustering methods, and OT optimization (Sinkhorn).  
4. **Method**: detailed math derivation of the objective, proof sketch for differentiability and stability claim(s), algorithm pseudocode, complexity analysis.  
5. **Theoretical results**: at least one proposition/lemma on stability of learned prototypes under small data perturbations or on gradient properties of entropic OT w.r.t. prototype locations. Include assumptions and proof sketches; full proofs can be in appendix.  
6. **Implementation details**: architecture, hyperparameters, initialization, stopping criteria.  
7. **Experiments**: datasets, baselines, evaluation protocol, main quantitative results (tables with ARI/NMI avg ± std), ablation studies, runtime/scalability.  
8. **Analyses**: visualization and qualitative discussion, failure modes, limitations.  
9. **Discussion & conclusion**: summarize findings, propose future extensions.  
10. **Reproducibility statement**: link to code, instructions to reproduce main results.  
11. **Appendix / supplementary**: detailed proofs, extra ablations, full hyperparameter tables, additional visuals.

**Formatting requirements:** prepare both a NeurIPS/JMLR style LaTeX template and a compact two‑column conference version for initial submission. Use BibTeX for references.

---

## Statistical tests and table formatting

- For each dataset, produce a table with mean ± std for ARI and NMI for each method.  
- Report pairwise significance vs the proposed method (p‑values) and mark significance in tables (e.g., bold for best, underline for second best).  
- Run Wilcoxon signed‑rank across seeds for each dataset; for multiple datasets run Friedman test and Nemenyi posthoc; produce a critical difference diagram.

---

## Compute budget & suggested ablation schedule

**Minimum compute to reproduce core claims:** 1 GPU with 16GB (e.g., NVIDIA V100), 8 CPU cores, 64GB RAM. Estimated time: ~48–120 hours for full benchmark depending on dataset scale.

**If more compute available:** run full image datasets end‑to‑end (train small CNN encoders), include CIFAR and larger scRNA sets.

**Phased schedule (12 weeks example):**
- Week 1–2: implement core OT module, unrolled Sinkhorn with tests; implement prototype module and optimizer; implement synthetic experiments.  
- Week 3–4: embedding network + image dataset pipelines; run baselines on synthetic and a few real datasets.  
- Week 5–6: full benchmarking (10 seeds), ablations on epsilon and masses.  
- Week 7–8: scalability variants (mini‑batch OT) and efficiency profiling.  
- Week 9: statistical tests and figure/table generation.  
- Week 10–11: write paper + supplementary; LaTeX compilation and figure polishing.  
- Week 12: prepare release artifact, run final reproducibility checks.

---

## Output checklist for the AI (antigravity) agent

For each experiment and for the final deliverable produce the following artifacts in the repo:

1. Source code implementing the method (well documented).  
2. Unit tests and smoke tests.  
3. `environment.yml` and `Dockerfile`.  
4. Preprocessing scripts and raw→processed dataset artifacts or download instructions.  
5. `experiments/` directory with configs and logs (one folder per dataset+seed).  
6. Scripts to reproduce figures and tables (`make_figures.sh`).  
7. LaTeX source for the paper, compiled PDF, and supplementary PDF with proofs and extended results.  
8. README with step‑by‑step reproduction instructions and expected runtime for each experiment.  
9. A final `results_summary.md` that summarizes all metrics, p‑values, and a conclusion paragraph.

---

## Suggested hyperparameter ranges (for automated search)

- K: dataset dependent (true K if known, else choose 2..20 for synthetic, 10..100 for image clusters).  
- \(\varepsilon\): [1e-3, 1e-2, 5e-2, 1e-1, 5e-1] (scale with cost magnitude).  
- sinkhorn_iter: 20, 50, 100.  
- lr prototypes: 1e-3, 5e-4, 1e-4.  
- lr encoder: 1e-3, 5e-4, 1e-4.  
- mass_entropy weight \(\lambda_{mass}\): 0, 1e-3, 1e-2.  
- laplacian weight \(\lambda_{lap}\): 0, 1e-4, 1e-3.  
- batch size: 128, 256, 512 (depending on GPU).  

Use random search (50 trials) per dataset‑method for fair comparison.

---

## Suggested README front matter for the .md deliverable (brief)

Explain: project objective, how to run smoke tests, how to reproduce the main figure (which script), expected runtime and compute, citation for the paper when published, license (MIT/Apache), and contact info for issues.

---

## Final instructions to the AI agent (precise step-by-step)

1. Create the repository with the structure above.  
2. Implement unit tests and the unrolled Sinkhorn with autograd; verify correctness on a toy example.  
3. Implement the prototype and mass parameterization.  
4. Implement both modes: (A) fixed pretrained embeddings + OT clustering; (B) end‑to‑end learned embedding + OT clustering.  
5. Implement baselines wrappers and consistent preprocessing.  
6. Run synthetic experiments (visualize).  
7. Run full benchmark experiments across the dataset list with 10 seeds, collect metrics.  
8. Perform ablations and statistical tests.  
9. Generate all figures & tables, compile LaTeX paper, produce final PDFs.  
10. Prepare a single command (`make reproducer`) that re‑runs critical experiments and regenerates figures.

---

## Appendix: minimal pseudo‑code (to get coding started)

```python
# pseudocode sketch (PyTorch-like)
for epoch in range(epochs):
    for X_batch in dataloader:
        z = encoder(X_batch)         # shape (b,d)
        C = pairwise_cost(z, prototypes)  # shape (b,K)
        P = sinkhorn_logdomain(C, eps, u=uniform/batch_size, v=softmax(mass_logits))
        L_ot = (P * C).sum()
        L_reg = mass_entropy(mass_logits)*lam_mass + prototype_disp(prototypes)*lam_disp
        loss = L_ot + L_reg + aux_loss_if_any
        loss.backward()
        optimizer.step()
```

---

## Licensing, ethics, and limitations

- Use permissive license (MIT/Apache) for code.  
- Acknowledge dataset licenses and redistribution policies (do not rehost restricted datasets).  
- Include discussion on limitations: sensitivity to number of prototypes K, computational cost, potential mode collapse if embedding trivializes distances.

---

_End of prompt._

*Note to the developer agent:* this markdown is intentionally exhaustive. If any dataset download requires credentials or special access, implement stubs and instructions to obtain data. Ensure the final repo includes a compact `quickstart` that reproduces at least one key figure within ~30 minutes on a single GPU.

