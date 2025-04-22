import numpy as np
import matplotlib.pyplot as plt
import os
import numba
from sklearn.mixture import GaussianMixture

# --- Settings ---
GROUP_REVS = 1  # Number of revolutions to average over
SYMMETRY_REVS = 26 / 5  # For 26/5 revolution symmetry
REFERENCE_NUM_POINTS = 200  # Used to convert phase0 to float phase

def compute_revolution_frequency(phase, phase_time, rev_step=2*np.pi):
    """
    Compute frequency based on time between full 2π phase revolutions.
    Returns:
        freq_times (np.ndarray): Times where speed is computed.
        freq_values (np.ndarray): Frequency values (Hz).
    """
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

        # Compute revolution-based frequency
        # Speed per 1 rev
        t1, s1 = compute_revolution_frequency(phase, phase_time)

        # Speed per 26/5 rev
        symmetry_step = SYMMETRY_REVS * 2 * np.pi
        t2, s2 = compute_revolution_frequency(phase, phase_time, rev_step=symmetry_step)
        s2 *= SYMMETRY_REVS  # Adjust for frequency units (Hz)
        # (Handled above as t1, s1 and t2, s2)

        # Moving average (window=5) of Per 26/5 rev speed
        window = 5
        s3 = np.convolve(s2, np.ones(window)/window, mode='valid')
        t3 = t2[window // 2 : -((window - 1) // 2)]

        duration = phase_time[-1] - phase_time[0]

        trace_configs = [
            (t1, s1, "Per Rev", "rev1", 0.2),
            (t2, s2, "Per 26/5 Rev", "rev26_5", 0.22),
            (t3, s3, "Smoothed 26/5 (5 pt MA)", "rev26", 0.22)
        ]

        # Fit GMM to each speed distribution and store peaks
        gmm_results = {}
        for s, tag in [(s1, "rev1"), (s2, "rev26_5"), (s3, "rev26")]:
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
            gmm_results[tag] = peaks

        for t, s, label, tag, sigma_factor in trace_configs:
            plt.figure(figsize=(12, 4))
            plt.plot(t, s, label=f"Original ({label})", alpha=0.3)
            sigma = sigma_factor * np.std(s)
            s_smooth = chi2_filter_njit_flat_steps(s, sigma)
            plt.plot(t, s_smooth, label=f"Chi2 σ={sigma_factor:.2f}×std", linewidth=1.5)
            # Plot GMM peaks
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
            print(f"✅ Saved {label} speed trace to: {out_path}")
            plt.close()

        # Plot normalized histogram (now normalize by max for plotting)
        plt.figure(figsize=(8, 5))
        counts1, bins1 = np.histogram(s1, bins=100)
        counts2, bins2 = np.histogram(s2, bins=100)
        counts3, bins3 = np.histogram(s3, bins=100)
        bin_width1 = bins1[1] - bins1[0]
        bin_width2 = bins2[1] - bins2[0]
        bin_width3 = bins3[1] - bins3[0]
        counts1 = counts1 / np.max(counts1)
        counts2 = counts2 / np.max(counts2)
        counts3 = counts3 / np.max(counts3)
        plt.bar(bins1[:-1], counts1, width=bin_width1, alpha=0.5, label="Per Rev")
        plt.bar(bins2[:-1], counts2, width=bin_width2, alpha=0.5, label="Per 26/5 Rev")
        plt.bar(bins3[:-1], counts3, width=bin_width3, alpha=0.5, label="Per 26 Rev")
        plt.xlabel("Speed (Hz)")
        plt.ylabel("Normalized Count (max=1)")
        plt.title("Speed Distribution")
        plt.legend()
        plt.tight_layout()

        hist_path = os.path.splitext(npz_path)[0] + "_speed_hist.png"
        plt.savefig(hist_path)
        print(f"✅ Saved speed histogram to: {hist_path}")
        plt.close()

    except Exception as e:
        print(f"Failed to analyze {npz_path}: {e}")

@numba.njit(parallel=True, fastmath=True)
def chi2_filter_njit_flat_steps(Y, sigma):
    N = len(Y)
    window_sizes = np.linspace(10, 1000, 10)
    window_sizes = window_sizes.astype(np.int32)
    smoothed_Y = np.zeros(N, dtype=np.float64)
    weighted_sum = np.zeros(N, dtype=np.float64)
    for w_idx in numba.prange(len(window_sizes)):
        w = window_sizes[w_idx]
        for i in range(N - w + 1):  # Sliding window over dataset
            Y_window = Y[i : i + w]
            Y_mean = np.sum(Y_window) / w
            chi2_values = np.sum((Y_window - Y_mean) ** 2) / (w * sigma**2)
            weight = np.exp(-chi2_values)
            smoothed_Y[i : i + w] += Y_mean * weight
            weighted_sum[i : i + w] += weight
    return smoothed_Y / weighted_sum


if __name__ == "__main__":
    # Change this path to your phase_data.npz
    path = "results/20250415/file6/phase_data.npz"
    analyze_single_file(path)