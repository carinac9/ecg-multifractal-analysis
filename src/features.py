import numpy as np
from scipy.signal import find_peaks


def extract_alpha0(alpha, f):
    return alpha[np.argmax(f)]


def extract_alpha_edges(alpha, f):
    idx_max = np.argmax(f)
    a1 = alpha[:idx_max].min()
    a2 = alpha[idx_max:].max()
    return a1, a2


def compute_slopes(alpha, f):
    idx = np.argmax(f)

    if idx > 1:
        left_fit = np.polyfit(alpha[:idx], f[:idx], 1)
        gamma1 = left_fit[0]
    else:
        gamma1 = 0.0

    if idx < len(alpha) - 1:
        right_fit = np.polyfit(alpha[idx:], f[idx:], 1)
        gamma2 = -right_fit[0]
    else:
        gamma2 = 0.0

    return gamma1, gamma2


def beat_detection(sig, fs, height=None):
    abs_sig = np.abs(sig)
    if height is None:
        height = np.mean(abs_sig) + np.std(abs_sig)

    peaks, _ = find_peaks(abs_sig, height=height, distance=int(0.4*fs))
    return peaks


def extract_beat(sig, peak_idx, beat_length=None):
    if beat_length is None:
        beat_length = len(sig) // 10

    start = max(0, peak_idx - beat_length // 2)
    end = min(len(sig), peak_idx + beat_length // 2)
    return sig[start:end]


def beat_replication_variability(sig, fs, alpha_orig, f_orig, num_beats=5):
    """
    Compute beat-to-beat variability of alpha_1 by replicating detected beats.

    Parameters:
    -----------
    sig : array-like
        Preprocessed ECG signal
    fs : int
        Sampling frequency
    alpha_orig : array-like
        Original multifractal exponents
    f_orig : array-like
        Original singularity spectrum
    num_beats : int, default=5
        Number of beats to replicate

    Returns:
    --------
    delta_a1 : float
        Variability index (original - mean replicated alpha_1)
    """
    try:
        from src.embedding import embed, svd_project
        from src.multifractal import multifractal_spectrum, subsample

        peaks = beat_detection(sig, fs)
        if len(peaks) < 2:
            return 0.0

        idx_max = np.argmax(f_orig)
        if idx_max == 0 or idx_max == len(alpha_orig) - 1:
            return 0.0

        a1_orig = alpha_orig[:idx_max].min()

        beat_alphas = []
        beats_to_check = min(num_beats, len(peaks))

        for peak in peaks[:beats_to_check]:
            beat = extract_beat(sig, peak, beat_length=int(0.3*fs))

            if len(beat) < 10:
                continue

            n_reps = len(sig) // len(beat) + 1
            sig_replicated = np.tile(beat, n_reps)[:len(sig)]

            M, TAU = 4, 198
            if len(sig_replicated) < (M-1)*TAU:
                continue

            try:
                Y = embed(sig_replicated, M, TAU)
                Y_svd = svd_project(Y)
                Y_sub = subsample(Y_svd)
                alpha_rep, f_rep = multifractal_spectrum(
                    Y_sub, np.linspace(-5, 5, 51))

                idx_max_rep = np.argmax(f_rep)
                if idx_max_rep > 0 and idx_max_rep < len(alpha_rep) - 1:
                    a1_rep = alpha_rep[:idx_max_rep].min()
                    beat_alphas.append(a1_rep)
            except:
                pass

        if not beat_alphas or len(beat_alphas) == 0:
            return 0.0

        delta_alpha1 = a1_orig - np.mean(beat_alphas)
        return float(delta_alpha1)

    except Exception:
        return 0.0


def extract_paper_features(alpha, f):
    a0 = extract_alpha0(alpha, f)
    a1, a2 = extract_alpha_edges(alpha, f)
    g1, g2 = compute_slopes(alpha, f)
    width = a2 - a1

    return a1, g1, a0, width
