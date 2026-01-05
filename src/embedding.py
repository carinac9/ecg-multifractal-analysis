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


def svd_project(Y, n_components=None):
    """
    SVD-based noise reduction and projection.
    Paper: "Use SVD before multifractal analysis for robust estimates"
    """
    if Y.ndim == 1:
        return Y.reshape(-1, 1)

    U, s, Vt = np.linalg.svd(Y, full_matrices=False)

    # Keep ~95% of variance (noise-robust)
    if n_components is None:
        cumvar = np.cumsum(s**2) / np.sum(s**2)
        n_components = np.argmax(cumvar >= 0.95) + 1
        n_components = min(n_components, len(s))

    # Project back
    Y_svd = U[:,
              :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]
    return Y_svd


def correlation_dimension_fast(Y, num_samples=1000):
    """
    Fast approximation of correlation dimension D2.
    Paper: "Verify nonlinearity via correlation dimension"
    """
    if len(Y) > num_samples:
        idx = np.random.choice(len(Y), num_samples, replace=False)
        Y = Y[idx]

    # Simple heuristic: variance of pairwise distances
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
    """
    Generate surrogate time series for nonlinearity test.
    Paper: "Generate surrogates to verify nonlinear deterministic dynamics"
    """
    if method == 'phase_randomized':
        # FFT randomization: preserve spectrum, randomize phase
        X = np.fft.fft(x)
        phases = np.random.rand(len(X)) * 2 * np.pi
        X_surr = np.abs(X) * np.exp(1j * phases)
        return np.real(np.fft.ifft(X_surr))
    elif method == 'amplitude_adjusted':
        # Preserve both spectrum and marginal distribution
        x_sorted = np.sort(x)
        ranks = np.argsort(np.argsort(np.random.randn(len(x))))
        return x_sorted[ranks]
    else:
        raise ValueError("Unknown surrogate method")
