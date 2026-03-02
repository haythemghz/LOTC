# MASTER EXECUTION PROMPT

## Project: Learned Optimal Transport Clustering (LOTC)

### Objective: Develop, Theoretically Strengthen, Experimentally Validate, and Publish in a Prestige Journal

You are an autonomous AI research system responsible for:

1.  Implementing LOTC from scratch.
2.  Strengthening it theoretically.
3.  Designing a rigorous experimental protocol.
4.  Running large-scale experiments.
5.  Writing a journal-level paper (TPAMI/JMLR quality).
6.  Preparing reproducible code and supplementary material.

This is NOT incremental research. The goal is to elevate LOTC into a
theoretically grounded and experimentally dominant clustering framework.

------------------------------------------------------------------------

# PRIORITY ROADMAP

## PRIORITY 1 --- Theoretical Strengthening (MANDATORY)

### 1.1 Formal Problem Definition

Given dataset X = {x_i} in R\^d, define clustering as:

min over prototypes C = {c_j} and transport plan P:

    L(C, P) = <P, D(X, C)> + ε H(P)

Subject to: - Row stochasticity constraints - Optional uniform column
constraints

Where: - D(X, C) = pairwise squared Euclidean distance matrix - H(P) =
entropic regularization - ε \> 0

------------------------------------------------------------------------

### 1.2 Required Theoretical Contributions

You MUST add at least TWO of the following:

1.  Prove convergence to k-means as ε → 0.
2.  Prove equivalence to a relaxed EM formulation.
3.  Provide stability bounds w.r.t ε.
4.  Provide Lipschitz continuity of prototype updates.
5.  Provide sample complexity analysis.
6.  Connect LOTC to Wasserstein barycenter theory formally.
7.  Show bias-variance tradeoff induced by entropy.

Formal theorem statements + proofs required.

------------------------------------------------------------------------

## PRIORITY 2 --- Algorithmic Design

### 2.1 Architecture

-   Implement Sinkhorn with log-domain stabilization.
-   Full GPU acceleration.
-   Automatic differentiation support.
-   Mini-batch OT for scalability.

### 2.2 Scalability Requirements

Test up to: - 100k samples - 512+ dimensional embeddings

Provide: - Time complexity analysis - Memory analysis - Empirical
runtime comparisons

------------------------------------------------------------------------

## PRIORITY 3 --- Experimental Protocol (Journal Level)

### 3.1 Datasets

Synthetic: - Gaussian mixtures - Non-convex manifolds - High-dimensional
noisy data

Real: - MNIST - CIFAR-10 - CIFAR-100 - STL-10 - 20 Newsgroups -
Reuters - ImageNet pretrained embeddings

Use fixed splits and reproducible seeds.

------------------------------------------------------------------------

### 3.2 Baselines (Mandatory)

Classical: - k-means++ - Spectral Clustering - Gaussian Mixture Models

Deep: - DEC - SCAN - DeepCluster

OT-based: - Existing Wasserstein clustering methods

------------------------------------------------------------------------

### 3.3 Metrics

External: - ACC - ARI - NMI

Internal: - Silhouette score - Davies-Bouldin

Stability: - Bootstrap consistency index - Seed variance analysis

Efficiency: - Runtime - GPU memory

------------------------------------------------------------------------

### 3.4 Ablation Studies

Required:

-   Sweep ε (log-scale)
-   Vary number of clusters
-   Initialization strategies
-   Remove entropy
-   Remove transport constraints

Produce full ablation tables and sensitivity plots.

------------------------------------------------------------------------

## PRIORITY 4 --- Statistical Validation

-   Perform 5--10 independent runs.
-   Report mean ± std.
-   Perform statistical significance tests (paired t-test).
-   Include confidence intervals.

------------------------------------------------------------------------

## PRIORITY 5 --- Failure Mode Analysis

-   Identify regimes where LOTC fails.
-   Analyze over-smoothing behavior.
-   Study cluster collapse cases.

Honest negative results required.

------------------------------------------------------------------------

# PAPER WRITING REQUIREMENTS

## Target: TPAMI / JMLR Level

Structure:

1.  Abstract (high-impact)
2.  Introduction (clear research gap)
3.  Related Work (position deeply within OT + clustering)
4.  Theoretical Section (formal)
5.  Algorithm Section
6.  Complexity Analysis
7.  Experimental Section
8.  Ablations
9.  Stability Analysis
10. Discussion
11. Limitations
12. Conclusion

Tone: - Formal - Precise - No hype - Clear contributions list

------------------------------------------------------------------------

# REPRODUCIBILITY PACKAGE

Must include:

-   Clean PyTorch code
-   Config files
-   Seed control
-   Dockerfile
-   README with full replication instructions

------------------------------------------------------------------------

# ACCEPTANCE OPTIMIZATION STRATEGY

To maximize probability of acceptance:

1.  Strong theoretical section.
2.  Clear superiority over k-means on structured datasets.
3.  Stability advantage demonstration.
4.  Scalability demonstration.
5.  Honest limitations section.

------------------------------------------------------------------------

# FINAL DELIVERABLES

1.  Complete codebase.
2.  All experiment logs.
3.  LaTeX paper ready for submission.
4.  Supplementary material.
5.  Reproducibility instructions.

------------------------------------------------------------------------

# SUCCESS CRITERIA

The work is considered submission-ready only if:

-   Results show consistent gains across multiple datasets.
-   At least one theoretical result is non-trivial.
-   Statistical significance is demonstrated.
-   Scalability is proven empirically.

End of execution specification.
