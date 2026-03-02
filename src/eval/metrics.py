import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the clustering accuracy using the Hungarian algorithm
    to find the optimal permutation between true and predicted labels.
    """
    from scipy.optimize import linear_sum_assignment
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.size == y_pred.size
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    return sum([w[row, col] for row, col in zip(row_ind, col_ind)]) / y_pred.size

def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Standard suite of clustering evaluation metrics.
    """
    return {
        'ACC': cluster_accuracy(y_true, y_pred),
        'ARI': adjusted_rand_score(y_true, y_pred),
        'NMI': normalized_mutual_info_score(y_true, y_pred)
    }
