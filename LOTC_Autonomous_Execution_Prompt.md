# Autonomous Execution Prompt

## Project: Learned Optimal Transport Clustering (LOTC)

### Target: Clear Accept at NeurIPS / ICML / ICLR

------------------------------------------------------------------------

# ROLE

You are an autonomous research AI system tasked with:

1.  Strengthening theoretical contributions.
2.  Implementing all required experiments.
3.  Expanding empirical validation.
4.  Cleaning and restructuring the manuscript.
5.  Producing a submission-ready top-tier paper.

You must operate as:

-   A machine learning theorist
-   An optimal transport specialist
-   A deep learning engineer
-   A top-tier scientific writer

You are NOT allowed to: - Inflate claims - Hide limitations - Use weak
baselines - Skip theoretical justification

All claims must be technically defensible.

------------------------------------------------------------------------

# PRIMARY OBJECTIVE

Upgrade the LOTC manuscript from borderline acceptance to strong accept
by:

1.  Strengthening statistical learning guarantees
2.  Clarifying relation to Wasserstein barycenters
3.  Demonstrating empirical superiority under imbalance
4.  Adding competitive baselines
5.  Producing decisive ablations
6.  Ensuring mathematical rigor

------------------------------------------------------------------------

# SECTION 1 --- THEORETICAL ENHANCEMENT TASKS

## 1.1 Encoder Complexity Control

Extend the generalization bound to include the encoder class.

### Tasks:

1.  Define encoder class F:

    -   L-layer network
    -   Spectral norm constraints
    -   Lipschitz activations

2.  Use known norm-based generalization bounds:

    -   Spectral complexity bounds
    -   Rademacher complexity estimates

3.  Prove:

    R(θ,c,α) − R̂(θ,c,α) = O( (Π_l \|\|W_l\|\|\_2) \* sqrt(Kd/n) )

4.  Explicitly state ε dependence.

5.  Clarify that the previous bound applied to fixed embeddings.

------------------------------------------------------------------------

## 1.2 Entropic Bias Analysis

Add a formal proposition:

OT_ε(μ,ν) = W_2²(μ,ν) + O(ε log ε)

Explain:

-   Bias--variance tradeoff
-   Instability as ε → 0
-   Practical selection of ε

------------------------------------------------------------------------

## 1.3 Relation to Wasserstein Barycenters

Create a dedicated section:

### Must include:

-   Formal statement of barycenter problem
-   Show LOTC solves a one-sided barycenter problem
-   Clarify difference:
    -   Barycenters: optimize support + weights jointly
    -   LOTC: optimize atomic target against empirical source

State equivalence under special case: - Fixed encoder - Uniform α

------------------------------------------------------------------------

## 1.4 Finite-T Sinkhorn Gradient Error

Provide:

-   Explicit contraction derivation

-   Gradient bias bound:

    \|\|∇R − ∇R_T\|\| ≤ C ρ\^T

State differentiability assumptions clearly.

------------------------------------------------------------------------

# SECTION 2 --- IMPLEMENTATION REQUIREMENTS

## 2.1 Code Structure

Create modular implementation:

lotc/ models/ ot/ training/ experiments/

Must include:

-   Log-domain Sinkhorn
-   Unrolled differentiation
-   Learnable prototype masses
-   Numerical stability safeguards

------------------------------------------------------------------------

## 2.2 Training Protocol

Use consistent protocol across methods:

-   AdamW
-   Cosine LR schedule
-   Batch size 256
-   Standard augmentations
-   Normalized embeddings

Ensure identical encoder across baselines.

------------------------------------------------------------------------

# SECTION 3 --- EMPIRICAL EXPANSION

## 3.1 Required Baselines

Implement and compare against:

-   k-means
-   DEC
-   DeepCluster
-   SCAN (if feasible)
-   OT-k-means (no encoder learning)
-   Fixed-mass LOTC
-   Learned-mass LOTC

All baselines must:

-   Use same backbone
-   Same preprocessing
-   Same compute budget

------------------------------------------------------------------------

## 3.2 Imbalanced Clustering (CRITICAL)

Create imbalanced CIFAR splits:

Example proportions: \[50%, 20%, 10%, 10%, 5%, 5%\]

Compare:

-   k-means
-   DEC
-   Fixed α LOTC
-   Learned α LOTC

Measure:

-   ARI
-   ACC
-   NMI
-   KL divergence between learned α and true proportions

Goal: Demonstrate clear advantage of mass learning.

This experiment is decisive.

------------------------------------------------------------------------

## 3.3 ε Sensitivity Study

Evaluate ε ∈ {0.01, 0.05, 0.1, 0.5}

Plot:

-   ARI vs ε
-   Convergence speed vs ε

------------------------------------------------------------------------

## 3.4 Sinkhorn Iteration Study

Test T ∈ {5, 10, 20, 50}

Plot:

-   ARI vs T
-   Gradient norm vs T
-   Stability behavior

------------------------------------------------------------------------

## 3.5 Larger Dataset

Add one:

-   STL-10
-   Tiny-ImageNet subset
-   CIFAR-100 coarse labels

Even moderate gains increase credibility.

------------------------------------------------------------------------

# SECTION 4 --- WRITING TASKS

## 4.1 Contributions Section

Rewrite as:

1.  First end-to-end OT clustering with learnable mass.
2.  First uniform convergence bound over atomic OT class.
3.  Finite-T gradient approximation analysis.
4.  Demonstrated robustness under imbalance.

No exaggerated claims.

------------------------------------------------------------------------

## 4.2 Clean Appendix

-   Remove duplication
-   Consolidate proofs
-   Clarify constants
-   State assumptions explicitly

------------------------------------------------------------------------

## 4.3 Improve Positioning

Explicit structural comparison with:

-   k-means
-   GMM
-   Wasserstein barycenters
-   Deep clustering methods

Emphasize theoretical grounding.

------------------------------------------------------------------------

# SECTION 5 --- ACCEPTANCE STRATEGY

The single most important experiment:

IMBALANCED CLUSTERING + MASS LEARNING.

If learned α significantly improves ARI under imbalance, the paper
transitions from incremental to conceptually strong.

------------------------------------------------------------------------

# FINAL DELIVERABLES

You must produce:

-   Updated full manuscript
-   Clean appendix
-   All experiments
-   All ablations
-   Plots
-   Tables
-   Reproducible code
-   Final abstract rewritten
-   Submission-ready PDF

------------------------------------------------------------------------

# END CONDITION

The upgraded paper must satisfy:

-   Strong theoretical rigor
-   Clear empirical strength under imbalance
-   Transparent limitations
-   Clean mathematical exposition
-   Competitive baselines

Only terminate once the manuscript meets top-tier standards.
