# LOTC Enhancement Action Plan

## Objective: Upgrade Paper from Weak Accept to Strong Accept at Top-Tier Venue

------------------------------------------------------------------------

# Priority Structure

-   🔴 P0 --- Critical (Must Fix for Strong Accept)
-   🟠 P1 --- High Impact Improvements
-   🟡 P2 --- Strengthening & Positioning
-   🟢 P3 --- Polish & Presentation

------------------------------------------------------------------------

# 🔴 P0 --- CRITICAL IMPROVEMENTS

## 1. Demonstrate Clear Empirical Superiority Over K-Means

### Problem

Current CIFAR-10 results show marginal improvements over K-Means (ARI
0.461 → 0.463). This is insufficient for top-tier impact.

### Required Actions

1.  Add larger and more complex benchmarks:
    -   CIFAR-100 (full 100 classes)
    -   ImageNet-100
    -   STL-10
    -   Tiny-ImageNet
    -   At least one real-world tabular dataset with strong imbalance
2.  Add harder settings:
    -   Severe imbalance (1:50, 1:100)
    -   Overlapping clusters
    -   Noisy labels
    -   Distribution shift (train/test split mismatch)
3.  Add robustness stress tests:
    -   Reduce SSL quality intentionally
    -   Add Gaussian noise to embeddings
    -   Vary batch size aggressively

### Target Outcome

LOTC should show ≥3--5% ARI improvement on at least one challenging
benchmark.

------------------------------------------------------------------------

## 2. Add Stronger Baselines

### Missing Comparisons

Include: - DeepCluster-v2 - DINO clustering - Balanced Softmax
clustering - Spectral clustering on SSL features - Spherical GMM -
OT-KMeans variants from literature

### Target Outcome

Show LOTC outperforming modern clustering pipelines, not only classical
ones.

------------------------------------------------------------------------

## 3. Clarify Novelty vs Prior OT Work

### Required Actions

1.  Add subsection: "Relation to Existing OT Clustering Methods"
2.  Explicitly compare to:
    -   Wasserstein dictionary learning
    -   Sinkhorn KMeans approximations
    -   OT barycenter clustering
3.  Add comparison table summarizing:
    -   Differentiable?
    -   Joint encoder learning?
    -   Mass learning?
    -   Convergence guarantees?

### Target Outcome

Remove overstatements and precisely position contribution.

------------------------------------------------------------------------

# 🟠 P1 --- HIGH IMPACT ENHANCEMENTS

## 4. Strengthen Theoretical Contributions

### A. Improve Bias--Variance Section

-   Provide empirical validation of ε tradeoff curve.
-   Add plot: ARI vs ε vs theoretical bound.

### B. Add Practical ε Selection Strategy

-   Derive heuristic based on Δ / ε contraction rate.
-   Provide rule-of-thumb formula.

### C. Add Landscape Insight

-   Empirically visualize loss surface slices.
-   Compare smoothness vs K-Means.

------------------------------------------------------------------------

## 5. Show Where LOTC Clearly Wins

### Suggested Demonstrations

1.  Extreme Imbalance Regime
2.  Non-convex Manifold Recovery at Scale
3.  Small Batch Regime Stability
4.  Prototype Interpretability (visual examples)

------------------------------------------------------------------------

# 🟡 P2 --- POSITIONING & FRAMING

## 6. Reframe Claims

Replace: - "Decisive dominance" - "First fully differentiable..."

With: - "Principled geometric alternative" - "Unified OT-based
framework"

Tone must match empirical magnitude.

------------------------------------------------------------------------

## 7. Add Conceptual Diagram

Add a single strong figure: - Data distribution - Prototype measure -
Transport plan - Mass adaptation

This improves accessibility and memorability.

------------------------------------------------------------------------

## 8. Improve Practical Relevance Section

Add subsection: - When to use LOTC vs KMeans - When imbalance matters -
Computational break-even analysis

------------------------------------------------------------------------

# 🟢 P3 --- POLISH & OPTIMIZATION

## 9. Tighten Theoretical Constants

-   Explicit constant tracking in generalization bounds.
-   Clarify spectral norm assumptions.
-   Discuss practical bounds on Λ.

------------------------------------------------------------------------

## 10. Add Failure Case Analysis

Provide visualizations of: - Entropic collapse - High-K smoothing - Poor
SSL initialization

Explain clearly why they occur.

------------------------------------------------------------------------

# Implementation Roadmap

## Week 1--2

-   Run expanded benchmark suite
-   Add stronger baselines
-   Perform imbalance stress tests

## Week 3

-   Add ε tradeoff empirical validation
-   Add loss landscape visualization
-   Strengthen novelty positioning

## Week 4

-   Rewrite introduction & claims
-   Add diagrams & improved framing
-   Final polish and ablations

------------------------------------------------------------------------

# Success Criteria Checklist

-   [ ] ≥1 dataset with clear performance margin
-   [ ] Modern SSL clustering baselines included
-   [ ] Claims toned to match evidence
-   [ ] Strong novelty positioning section
-   [ ] Empirical validation of theory
-   [ ] Clear practical guidance section

------------------------------------------------------------------------

# Final Goal

Transform LOTC from:

> "Elegant OT refinement of KMeans"

Into:

> "A principled and empirically necessary OT-based clustering framework
> for imbalanced and geometrically complex data."
