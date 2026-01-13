import numpy as np
from sklearn.metrics import mutual_info_score


def optimal_tau(x, max_lag=200):
    mis = [mutual_info_score(x[:-lag], x[lag:]) for lag in range(1, max_lag)]
    return int(np.argmin(mis))


def embed(x, M, tau):
    N = len(x) - (M - 1) * tau
    return np.array([x[i:i + M * tau:tau] for i in range(N)])


def svd_project(Y, n_components=None):
    if Y.ndim == 1:
        return Y.reshape(-1, 1)

    U, s, Vt = np.linalg.svd(Y, full_matrices=False)

    if n_components is None:
        cumvar = np.cumsum(s**2) / np.sum(s**2)
        n_components = np.argmax(cumvar >= 0.95) + 1
        n_components = min(n_components, len(s))

    Y_svd = U[:,
              :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]
    return Y_svd


def correlation_dimension_fast(Y, num_samples=1000):
    if len(Y) > num_samples:
        idx = np.random.choice(len(Y), num_samples, replace=False)
        Y = Y[idx]

    N = min(len(Y), 100)
    if N < 2:
        return 1.0

    idx = np.random.choice(len(Y), N, replace=False)
    sample = Y[idx]

    dists = []
    for i in range(len(sample)):
        for j in range(i+1, len(sample)):
            d = np.sqrt(np.sum((sample[i] - sample[j])**2))
            dists.append(d)

    return np.std(dists) if dists else 1.0


def surrogate_signal(x, method='phase_randomized'):
    if method == 'phase_randomized':
        X = np.fft.fft(x)
        phases = np.random.rand(len(X)) * 2 * np.pi
        X_surr = np.abs(X) * np.exp(1j * phases)
        return np.real(np.fft.ifft(X_surr))
    elif method == 'amplitude_adjusted':
        x_sorted = np.sort(x)
        ranks = np.argsort(np.argsort(np.random.randn(len(x))))
        return x_sorted[ranks]
    else:
        raise ValueError("Unknown surrogate method")
