import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numba

# === Load data ===
csv_path = "results_backup/2025.05.23 patricia ox/files3/stepfit_speed_trace/itot_speed_trace_stepfit.csv"
df = pd.read_csv(csv_path)
t = df["Time (s)"].values
speed = df["Speed (Hz)"].values
smoothed = df["Smoothed Speed (Hz)"].values

# === Chi2 filtering ===
@numba.njit(parallel=True, fastmath=True)
def chi2_filter_njit_flat_steps(Y, sigma):
    N = len(Y)
    window_sizes = np.linspace(10, 1000, 10).astype(np.int32)
    smoothed_Y = np.zeros(N, dtype=np.float64)
    weighted_sum = np.zeros(N, dtype=np.float64)
    for w_idx in numba.prange(len(window_sizes)):
        w = window_sizes[w_idx]
        for i in range(N - w + 1):
            Y_window = Y[i : i + w]
            Y_mean = np.sum(Y_window) / w
            chi2_values = np.sum((Y_window - Y_mean) ** 2) / (w * sigma**2)
            weight = np.exp(-chi2_values)
            smoothed_Y[i : i + w] += Y_mean * weight
            weighted_sum[i : i + w] += weight
    result = np.empty_like(smoothed_Y)
    for i in range(N):
        if weighted_sum[i] > 0:
            result[i] = smoothed_Y[i] / weighted_sum[i]
        else:
            result[i] = np.nan  # or 0.0, depending on preference
    return result

sigma_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
window_sets = [
    np.linspace(500, 3000, 10).astype(np.int32),
]

for window_sizes in window_sets:
    for sigma in sigma_values:
        @numba.njit(parallel=True, fastmath=True)
        def chi2_custom(Y, sigma):
            N = len(Y)
            smoothed_Y = np.zeros(N, dtype=np.float64)
            weighted_sum = np.zeros(N, dtype=np.float64)
            for w_idx in numba.prange(len(window_sizes)):
                w = window_sizes[w_idx]
                for i in range(N - w + 1):
                    Y_window = Y[i : i + w]
                    Y_mean = np.sum(Y_window) / w
                    chi2_values = np.sum((Y_window - Y_mean) ** 2) / (w * sigma**2)
                    weight = np.exp(-chi2_values)
                    smoothed_Y[i : i + w] += Y_mean * weight
                    weighted_sum[i : i + w] += weight
            result = np.empty_like(smoothed_Y)
            for i in range(N):
                if weighted_sum[i] > 0:
                    result[i] = smoothed_Y[i] / weighted_sum[i]
                else:
                    result[i] = np.nan
            return result

        pad_length = 10000
        padded_smoothed = np.concatenate([np.full(pad_length, smoothed[0]), smoothed])
        padded_filtered = chi2_custom(padded_smoothed, sigma)
        filtered = padded_filtered[pad_length:]

        # === Step Fitting ===
        def get_variance(data, inter, pos):
            attempt = inter[:np.searchsorted(inter, pos)] + [pos] + inter[np.searchsorted(inter, pos):]
            delta = list(data[:attempt[0]] - np.mean(data[:attempt[0]]))
            for i in range(len(attempt) - 1):
                seg = data[attempt[i]:attempt[i + 1]]
                if len(seg) > 0:
                    delta += list(seg - np.mean(seg))
            delta += list(data[attempt[-1]:] - np.mean(data[attempt[-1]:]))
            return np.mean(np.array(delta) ** 2.)

        def get_pos(data, inter, res=50):
            variance = np.inf
            best_pos = None
            for pos in range(res, len(data) - res, res):
                if pos not in inter and 0 < pos < len(data):
                    v = get_variance(data, inter, pos)
                    if v < variance:
                        variance = v
                        best_pos = pos
            if best_pos is not None:
                inter = inter[:np.searchsorted(inter, best_pos)] + [best_pos] + inter[np.searchsorted(inter, best_pos):]
            return best_pos, inter, variance

        def get_int_av(data, res=5, th=10, limit_ratio=0.99):
            inter = [0]
            variance = 100
            v_random = 200
            real = 10
            max_len = len(data)
            step_count = 0
            while variance / v_random < limit_ratio:
                v_random = np.mean([
                    get_variance(data, inter, np.random.randint(0, max_len))
                    for _ in range(real)
                ])
                best_pos, inter, variance = get_pos(data, inter, res=res)
                step_count += 1
                if best_pos is None:
                    break
            inter = sorted(set(inter))
            if inter[-1] != max_len - 1:
                inter.append(max_len - 1)
            step_fitted = np.zeros_like(data)
            for i in range(len(inter) - 1):
                start, end = inter[i], inter[i + 1]
                if end > start:
                    avg = np.mean(data[start:end])
                    step_fitted[start:end] = avg
            return step_fitted

        step_result = get_int_av(filtered, res=5, th=10)

        # === Plot with step ===
        plt.figure(figsize=(12, 4))
        plt.plot(t, speed, label="Raw Speed", alpha=0.3)
        plt.plot(t, smoothed, label="Savgol Smoothed", alpha=0.6)
        plt.plot(t, filtered, label=f"Chi2 (σ={sigma}, win={window_sizes[0]}–{window_sizes[-1]})", linewidth=2)
        plt.plot(t, step_result, label="Step-Fitted", linestyle='--', linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (Hz)")
        plt.title(f"Chi2 Filter + Step Fit σ={sigma}, window={window_sizes[0]}–{window_sizes[-1]}")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()