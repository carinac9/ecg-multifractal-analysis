import numpy as np


def extract_alpha0(alpha, f):
    return alpha[np.argmax(f)]


def extract_alpha_edges(alpha, f):
    idx_max = np.argmax(f)
    a1 = alpha[:idx_max].min()
    a2 = alpha[idx_max:].max()
    return a1, a2


def compute_slopes(alpha, f):
    idx = np.argmax(f)
    left_fit = np.polyfit(alpha[:idx], f[:idx], 1)
    right_fit = np.polyfit(alpha[idx:], f[idx:], 1)
    gamma1 = left_fit[0]
    gamma2 = -right_fit[0]
    return gamma1, gamma2
