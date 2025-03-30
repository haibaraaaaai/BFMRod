import numpy as np

def normalize_signals(data, method="percentile", percentile=95):
    normalized = np.zeros_like(data)
    for i in range(data.shape[1]):
        channel = data[:, i]
        if method == "max":
            scale = np.max(channel)
        elif method == "mean":
            scale = np.mean(channel)
        elif method == "median":
            scale = np.median(channel)
        elif method == "percentile":
            scale = np.percentile(channel, percentile)
        else:
            raise ValueError("Unsupported normalization method")

        normalized[:, i] = channel / scale if scale != 0 else channel
    return normalized