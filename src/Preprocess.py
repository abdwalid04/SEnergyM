import numpy as np

def undo_normalize(signal, mean, std):
    return signal * std + mean

def _quantile_signal(signal, window_size, quantile=.5):
    quantile_signal = np.empty_like(signal)
    for i in range(0, len(signal), window_size):
        window = signal[i:i + window_size]
        q = np.quantile(window, quantile)
        quantile_signal[i:i + window_size] = q
    return quantile_signal