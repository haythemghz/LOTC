"""
Baseline clustering method wrappers for fair comparison.

Each wrapper follows a consistent interface:
- ``.fit(X)`` — fit on data.
- ``.predict(X)`` — return cluster labels.
- ``.labels_`` — labels from the last ``.fit()`` call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture


class BaselineWrapper(ABC):
    """Abstract baseline interface."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.labels_: np.ndarray | None = None

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaselineWrapper":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_


class KMeansBaseline(BaselineWrapper):
    def __init__(self, n_clusters: int = 5, seed: int = 42, **kwargs):
        super().__init__(name="KMeans")
        self.model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)

    def fit(self, X: np.ndarray) -> "KMeansBaseline":
        self.model.fit(X)
        self.labels_ = self.model.labels_
        return self


class GMMBaseline(BaselineWrapper):
    def __init__(self, n_components: int = 5, seed: int = 42, **kwargs):
        super().__init__(name="GMM")
        self.model = GaussianMixture(
            n_components=n_components, random_state=seed, n_init=5
        )

    def fit(self, X: np.ndarray) -> "GMMBaseline":
        self.model.fit(X)
        self.labels_ = self.model.predict(X)
        return self


class AgglomerativeBaseline(BaselineWrapper):
    def __init__(self, n_clusters: int = 5, linkage: str = "ward", **kwargs):
        super().__init__(name=f"Agglomerative({linkage})")
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

    def fit(self, X: np.ndarray) -> "AgglomerativeBaseline":
        self.model.fit(X)
        self.labels_ = self.model.labels_
        return self


class DBSCANBaseline(BaselineWrapper):
    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs):
        super().__init__(name="DBSCAN")
        self.model = DBSCAN(eps=eps, min_samples=min_samples)

    def fit(self, X: np.ndarray) -> "DBSCANBaseline":
        self.model.fit(X)
        self.labels_ = self.model.labels_
        return self


class HDBSCANBaseline(BaselineWrapper):
    def __init__(self, min_cluster_size: int = 15, **kwargs):
        super().__init__(name="HDBSCAN")
        self.min_cluster_size = min_cluster_size

    def fit(self, X: np.ndarray) -> "HDBSCANBaseline":
        try:
            from sklearn.cluster import HDBSCAN as SkHDBSCAN
            model = SkHDBSCAN(min_cluster_size=self.min_cluster_size)
        except ImportError:
            import hdbscan
            model = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        model.fit(X)
        self.labels_ = model.labels_
        return self


class SpectralBaseline(BaselineWrapper):
    def __init__(self, n_clusters: int = 5, seed: int = 42, **kwargs):
        super().__init__(name="Spectral")
        self.model = SpectralClustering(
            n_clusters=n_clusters, random_state=seed, affinity="rbf"
        )

    def fit(self, X: np.ndarray) -> "SpectralBaseline":
        self.model.fit(X)
        self.labels_ = self.model.labels_
        return self


# ---- Factory ----

BASELINES = {
    "kmeans": KMeansBaseline,
    "gmm": GMMBaseline,
    "agglomerative": AgglomerativeBaseline,
    "dbscan": DBSCANBaseline,
    "hdbscan": HDBSCANBaseline,
    "spectral": SpectralBaseline,
}


def get_baseline(name: str, **kwargs) -> BaselineWrapper:
    """Instantiate a baseline by name."""
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(BASELINES.keys())}")
    return BASELINES[name](**kwargs)


def get_all_baselines(n_clusters: int = 5, seed: int = 42) -> list[BaselineWrapper]:
    """Create all classical baselines with default settings."""
    return [
        KMeansBaseline(n_clusters=n_clusters, seed=seed),
        GMMBaseline(n_components=n_clusters, seed=seed),
        AgglomerativeBaseline(n_clusters=n_clusters),
        DBSCANBaseline(),
        HDBSCANBaseline(),
        SpectralBaseline(n_clusters=n_clusters, seed=seed),
    ]
