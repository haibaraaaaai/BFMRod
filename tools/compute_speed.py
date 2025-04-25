import numpy as np
import matplotlib.pyplot as plt
import os
import numba
from sklearn.mixture import GaussianMixture

# --- Settings ---
GROUP_REVS = 1  # Number of revolutions to average over
REFERENCE_NUM_POINTS = 200  # Used to convert phase0 to float phase

def compute_revolution_frequency(phase, phase_time, rev_step=2*np.pi):
    step = rev_step
    max_phase = phase[-1]
    thresholds = np.arange(step, max_phase, step)

    rev_times = []
    for threshold in thresholds:
        idx = np.searchsorted(phase, threshold)
        if idx >= len(phase_time):
            break
        rev_times.append(phase_time[idx])

    rev_times = np.array([phase_time[0]] + rev_times)

    freq_times = []
    freq_values = []
    for i in range(len(rev_times) - 1):
        t_start = rev_times[i]
        t_end = rev_times[i + 1]
        freq = 1 / (t_end - t_start)

        mid1 = i + (1 - 1) // 2
        mid2 = i + (1 + 1) // 2
        freq_time = (rev_times[mid1] + rev_times[mid2]) / 2

        freq_times.append(freq_time)
        freq_values.append(freq)

    return np.array(freq_times), np.array(freq_values)

def analyze_single_file(npz_path):
    try:
        data = np.load(npz_path)
        phase0 = data["phase0"]
        phase_time = data["phase_time"]

        # Convert to continuous float phase
        phase = phase0 / REFERENCE_NUM_POINTS * 2 * np.pi
        phase = np.unwrap(phase)

        # Speed per 1 rev
        t1, s1 = compute_revolution_frequency(phase, phase_time)

        trace_configs = [
            (t1, s1, "Per Rev", "rev1", 0.2),
        ]

        # GMM fitting (with masking > 2000 Hz)
        gmm_results = {}
        s = s1[s1 < 2000]
        s_reshaped = s.reshape(-1, 1)
        best_gmm = None
        best_bic = np.inf
        for n in range(1, 6):
            gmm = GaussianMixture(n_components=n, random_state=0).fit(s_reshaped)
            bic = gmm.bic(s_reshaped)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        peaks = np.sort(best_gmm.means_.flatten())
        gmm_results["rev1"] = peaks

        # Plot time traces with GMM peaks and chi2 smoothing
        for t, s, label, tag, sigma_factor in trace_configs:
            mask = s < 2000
            t = t[mask]
            s = s[mask]
            plt.figure(figsize=(12, 4))
            plt.plot(t, s, label=f"Original ({label})", alpha=0.3)
            sigma = sigma_factor * np.std(s)
            s_smooth = chi2_filter_njit_flat_steps(s, sigma)
            plt.plot(t, s_smooth, label=f"Chi2 σ={sigma_factor:.2f}×std", linewidth=1.5)
            for peak in gmm_results.get(tag, []):
                plt.axhline(peak, linestyle="--", color="gray", alpha=0.5)
            plt.xlabel("Time (s)")
            plt.ylabel("Speed (Hz)")
            plt.title(f"{label} Speed and Chi2 Fit")
            plt.ylim(0, 1.2 * np.max(s))
            plt.legend()
            plt.tight_layout()

            out_path = os.path.splitext(npz_path)[0] + f"_speed_trace_{tag}.png"
            plt.savefig(out_path)
            print(f"Saved {label} speed trace to: {out_path}")
            plt.close()

        # Plot normalized histogram
        plt.figure(figsize=(8, 5))
        counts1, bins1 = np.histogram(s1[s1 < 2000], bins=100)
        bin_width1 = bins1[1] - bins1[0]
        counts1 = counts1 / np.max(counts1)
        plt.bar(bins1[:-1], counts1, width=bin_width1, alpha=0.5, label="Per Rev")
        plt.xlabel("Speed (Hz)")
        plt.ylabel("Normalized Count (max=1)")
        plt.title("Speed Distribution")
        plt.legend()
        plt.tight_layout()

        hist_path = os.path.splitext(npz_path)[0] + "_speed_hist.png"
        plt.savefig(hist_path)
        print(f"Saved speed histogram to: {hist_path}")
        plt.close()

    except Exception as e:
        print(f"Failed to analyze {npz_path}: {e}")

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
    return smoothed_Y / weighted_sum

if __name__ == "__main__":
    # Update this path to your actual file
    path = "results_backup/2025.04.16 patricia/file8/phase_data.npz"
    analyze_single_file(path)
