import numpy as np
from scipy.signal import find_peaks


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


def beat_detection(sig, fs, height=None):
    """
    Detect QRS complexes (R peaks) in ECG signal.
    Paper: "Extract individual beats for beat-replication analysis"
    """
    # Find peaks in absolute signal
    abs_sig = np.abs(sig)
    if height is None:
        height = np.mean(abs_sig) + np.std(abs_sig)

    peaks, _ = find_peaks(abs_sig, height=height, distance=int(0.4*fs))
    return peaks


def extract_beat(sig, peak_idx, beat_length=None):
    """Extract single beat around peak."""
    if beat_length is None:
        beat_length = len(sig) // 10  # ~100ms at typical fs

    start = max(0, peak_idx - beat_length // 2)
    end = min(len(sig), peak_idx + beat_length // 2)
    return sig[start:end]


def beat_replication_variability(sig, fs, alpha_orig, f_orig):
    """
    Beat-replication variability check (paper's self-referenced test).
    Paper: "Extract beats, replicate to full length, compute α₁ variability"

    Returns: δα₁ = α₁(original) - mean(α₁(replicated_beats))
    Healthy → small δα₁; Unhealthy → large δα₁
    """
    try:
        from src.embedding import embed, svd_project
        from src.multifractal import multifractal_spectrum, subsample

        # Detect beats
        peaks = beat_detection(sig, fs)
        if len(peaks) < 2:
            return 0.0  # Not enough beats

        # Get original α₁
        idx_max = np.argmax(f_orig)
        a1_orig = alpha_orig[:idx_max].min()

        # Extract individual beats
        beat_alphas = []
        for peak in peaks[:5]:  # Use up to 5 beats
            beat = extract_beat(sig, peak, beat_length=int(0.3*fs))

            if len(beat) < 10:
                continue

            # Replicate beat to full signal length
            n_reps = len(sig) // len(beat) + 1
            sig_replicated = np.tile(beat, n_reps)[:len(sig)]

            # Compute multifractal for replicated beat
            M, TAU = 4, 198
            if len(sig_replicated) < (M-1)*TAU:
                continue

            Y = embed(sig_replicated, M, TAU)
            Y_svd = svd_project(Y)
            Y_sub = subsample(Y_svd)
            alpha_rep, f_rep = multifractal_spectrum(
                Y_sub, np.linspace(-5, 5, 51))

            # Extract α₁ from replicated
            idx_max_rep = np.argmax(f_rep)
            a1_rep = alpha_rep[:idx_max_rep].min()
            beat_alphas.append(a1_rep)

        if not beat_alphas:
            return 0.0

        # δα₁ = variability metric
        delta_alpha1 = a1_orig - np.mean(beat_alphas)
        return delta_alpha1

    except Exception:
        return 0.0  # Safe fallback


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
