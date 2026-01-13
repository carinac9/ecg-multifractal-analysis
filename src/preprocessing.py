import numpy as np
from scipy.signal import detrend, butter, filtfilt


def preprocess_ecg(x, fs):
    x = detrend(x)

    b, a = butter(4, 40/(fs/2), btype='low')
    x = filtfilt(b, a, x)

    x = (x - np.mean(x)) / np.std(x)

    return x
