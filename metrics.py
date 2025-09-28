import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist, squareform

def compute_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def compute_auroc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)

def compute_aupr(y_true, y_scores):
    return average_precision_score(y_true, y_scores)

def compute_wasserstein_distance(real_embeddings, synth_embeddings):
    distances = []
    for i in range(real_embeddings.shape[1]):
        dist = wasserstein_distance(real_embeddings[:, i], synth_embeddings[:, i])
        distances.append(dist)
    return np.mean(distances)

def compute_kl_divergence(q_samples, p_samples=None):
    if p_samples is None:
        p_samples = torch.randn_like(q_samples)
    q_samples = q_samples.detach().cpu().numpy()
    p_samples = p_samples.detach().cpu().numpy()
    # Estimate KL via kNN (simplified)
    from sklearn.neighbors import NearestNeighbors
    n = q_samples.shape[0]
    k = min(5, n-1)
    nbrs_p = NearestNeighbors(n_neighbors=k).fit(p_samples)
    nbrs_q = NearestNeighbors(n_neighbors=k).fit(q_samples)
    dist_p, _ = nbrs_p.kneighbors(q_samples)
    dist_q, _ = nbrs_q.kneighbors(q_samples)
    kl = (np.log(dist_p[:, -1] / dist_q[:, -1]) + np.log(n / (n - 1))).mean()
    return kl