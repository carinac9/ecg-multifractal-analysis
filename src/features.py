import numpy as np


def extract_alpha0(alpha, f):
    """α₀: α at maximum f(α). Represents most probable singularity."""
    return alpha[np.argmax(f)]


def extract_alpha_edges(alpha, f):
    """
    Extract α₁ (left edge) and α₂ (right edge).
    Paper: "α₁ is most reliable; α₂ is numerically unstable - use with caution"
    """
    idx_max = np.argmax(f)
    a1 = alpha[:idx_max].min()
    a2 = alpha[idx_max:].max()
    return a1, a2


def compute_slopes(alpha, f):
    """
    Compute γ₁ (left slope) and γ₂ (right slope).
    Paper: "γ₁ is strong discriminator; γ₂ is less robust"
    """
    idx = np.argmax(f)

    # Left side (dense regions - more stable)
    if idx > 1:
        left_fit = np.polyfit(alpha[:idx], f[:idx], 1)
        gamma1 = left_fit[0]
    else:
        gamma1 = 0.0

    # Right side (sparse regions - less stable)
    if idx < len(alpha) - 1:
        right_fit = np.polyfit(alpha[idx:], f[idx:], 1)
        gamma2 = -right_fit[0]
    else:
        gamma2 = 0.0

    return gamma1, gamma2


def extract_paper_features(alpha, f):
    """
    Extract ONLY the most discriminative features per paper.
    Paper: "Use α₁, γ₁, α₀ for clinical-length signals. De-emphasize α₂."

    Returns: (α₁, γ₁, α₀, width)
    These are the best discriminators for healthy vs unhealthy.
    """
    a0 = extract_alpha0(alpha, f)
    a1, a2 = extract_alpha_edges(alpha, f)
    g1, g2 = compute_slopes(alpha, f)
    width = a2 - a1

    # Primary features per paper
    return a1, g1, a0, width
