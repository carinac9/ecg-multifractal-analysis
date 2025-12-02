import numpy as np


def subsample(Y, n=5000):
    """Subsample attractor points uniformly."""
    step = max(1, len(Y) // n)
    return Y[::step][:n]


def multifractal_spectrum(Y, q_vals, num_boxes=50):
    """Simplified Harikrishnan multifractal spectrum."""
    # Basic validation
    if Y is None or len(Y) == 0:
        alpha = np.zeros_like(q_vals, dtype=float)
        f_alpha = np.zeros_like(q_vals, dtype=float)
        f_alpha[len(f_alpha) // 2] = 1.0
        return alpha, f_alpha

    # normalize per-dimension, project to 1D
    denom = (Y.max(axis=0) - Y.min(axis=0)) + 1e-12
    Y_norm = (Y - Y.min(axis=0)) / denom

    # project to 1D (first dimension)
    try:
        Y1 = Y_norm[:, 0]
    except Exception:
        alpha = np.zeros_like(q_vals, dtype=float)
        f_alpha = np.zeros_like(q_vals, dtype=float)
        f_alpha[len(f_alpha) // 2] = 1.0
        return alpha, f_alpha

    # histogram-based probability estimate
    hist, _ = np.histogram(Y1, bins=num_boxes, density=True)
    total = np.sum(hist)
    if total <= 0:
        p = np.ones_like(hist) / len(hist)
    else:
        p = hist / total

    p = np.where(p <= 0, 1e-12, p)

    tau_q = []
    for q in q_vals:
        if np.isclose(q, 1.0):
            tau = -np.sum(p * np.log(p))
        else:
            s = np.sum(p ** q)
            s = max(s, 1e-12)
            tau = np.log(s) / np.log(1.0 / float(num_boxes))
        tau_q.append(tau)

    tau_q = np.array(tau_q, dtype=float)

    if np.all(np.isfinite(tau_q)):
        alpha = np.gradient(tau_q, q_vals)
        f_alpha = q_vals * alpha - tau_q
    else:
        alpha = np.zeros_like(q_vals, dtype=float)
        f_alpha = np.zeros_like(q_vals, dtype=float)
        f_alpha[len(f_alpha) // 2] = 1.0

    return alpha, f_alpha
