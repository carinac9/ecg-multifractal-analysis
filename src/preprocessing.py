import numpy as np
from scipy.signal import detrend, butter, filtfilt


def preprocess_ecg(x, fs):
    """Detrend, low-pass filter and normalize an ECG signal."""
    x = detrend(x)

    # 40 Hz Butterworth filter
    b, a = butter(4, 40/(fs/2), btype='low')
    x = filtfilt(b, a, x)

    # Z-score normalization
    x = (x - np.mean(x)) / np.std(x)

    return x
