import numpy as np
import matplotlib.pyplot as plt
import os

# --- Settings ---
GROUP_REVS = 1  # Number of revolutions to average over


def compute_revolution_frequency(phase, phase_time, group_revs=GROUP_REVS):
    """
    Compute frequency based on time between full 2π phase revolutions.
    Returns:
        freq_times (np.ndarray): Times where speed is computed.
        freq_values (np.ndarray): Frequency values (Hz).
    """
    step = 2 * np.pi
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
    for i in range(len(rev_times) - group_revs):
        t_start = rev_times[i]
        t_end = rev_times[i + group_revs]
        freq = group_revs / (t_end - t_start)

        if group_revs % 2 == 0:
            mid = i + group_revs // 2
            freq_time = rev_times[mid]
        else:
            mid1 = i + (group_revs - 1) // 2
            mid2 = i + (group_revs + 1) // 2
            freq_time = (rev_times[mid1] + rev_times[mid2]) / 2

        freq_times.append(freq_time)
        freq_values.append(freq)

    return np.array(freq_times), np.array(freq_values)


def analyze_single_file(npz_path):
    try:
        data = np.load(npz_path)
        phase = data["phase"]
        phase_time = data["phase_time"]

        # Compute revolution-based frequency
        t, s = compute_revolution_frequency(phase, phase_time)
        duration = phase_time[-1] - phase_time[0]

        # Plot time trace
        plt.figure(figsize=(12, 5))
        plt.plot(t, s, label="Speed (Hz)", linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (Hz)")
        plt.title("Instantaneous Frequency (Revolution-Based)")
        plt.tight_layout()

        timeplot_path = os.path.splitext(npz_path)[0] + "_speed_trace.png"
        plt.savefig(timeplot_path)
        print(f"✅ Saved speed trace to: {timeplot_path}")
        plt.close()

        # Plot normalized histogram
        plt.figure(figsize=(8, 5))
        weights = np.ones_like(s) / duration
        plt.hist(s, bins=100, weights=weights, alpha=0.8)
        plt.xlabel("Speed (Hz)")
        plt.ylabel("Normalized Count (/s)")
        plt.title("Speed Distribution")
        plt.tight_layout()

        hist_path = os.path.splitext(npz_path)[0] + "_speed_hist.png"
        plt.savefig(hist_path)
        print(f"✅ Saved speed histogram to: {hist_path}")
        plt.close()

    except Exception as e:
        print(f"Failed to analyze {npz_path}: {e}")


if __name__ == "__main__":
    # Change this path to your phase_data.npz
    path = "results backup/2025.03.20 patricia/file16/phase_data.npz"
    analyze_single_file(path)
