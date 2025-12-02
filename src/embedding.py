import numpy as np
from sklearn.metrics import mutual_info_score


def optimal_tau(x, max_lag=200):
    """Compute delay using first minimum of mutual information."""
    mis = [mutual_info_score(x[:-lag], x[lag:]) for lag in range(1, max_lag)]
    return int(np.argmin(mis))


def embed(x, M, tau):
    """Delay-coordinate embedding."""
    N = len(x) - (M - 1) * tau
    return np.array([x[i:i + M * tau:tau] for i in range(N)])
