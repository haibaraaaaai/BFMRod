import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# --- Settings ---
GROUP_REVS = 1
REFERENCE_NUM_POINTS = 200
SMOOTH_WINDOWS = [5, 11, 21, 31, 51, 71, 91, 121, 151, 201]  # must be odd

def compute_revolution_frequency(phase, phase_time, group_revs=GROUP_REVS):
    step = 2 * np.pi
    max_phase = phase[-1]
    thresholds = np.arange(step, max_phase, step)

    rev_times = []
    for threshold in thresholds:
        idx = np.searchsorted(phase, threshold)
        if idx >= len(phase_time):
            break
        rev_times.append(phase_time[idx])

    rev_times = np.array(rev_times)

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
        phase0 = data["phase0"]
        phase_time = data["phase_time"]

        # Unwrap full-scale phase first
        phase = phase0 / 200 * 2 * np.pi
        phase = np.unwrap(phase)

        if "harmonic2x" in npz_path.lower():
            phase *= 0.5  # Apply harmonic correction after unwrapping

        std_list = []
        win_labels = []
        speed_profiles = {}

        for win in SMOOTH_WINDOWS:
            if win >= len(phase):
                continue
            smoothed = savgol_filter(phase, window_length=win, polyorder=3)
            t, s = compute_revolution_frequency(smoothed, phase_time)
            std_list.append(np.std(s))
            win_labels.append(win)
            speed_profiles[win] = (t, s)

        # Plot STD vs smoothing window
        plt.figure(figsize=(6, 4))
        plt.plot(win_labels, std_list, marker="o")
        plt.xlabel("Smoothing Window (samples)")
        plt.ylabel("Speed Std (Hz)")
        plt.title("Std vs. Smoothing")
        plt.tight_layout()
        std_plot = os.path.splitext(npz_path)[0] + "_std_vs_smooth.png"
        plt.savefig(std_plot)
        print(f"✅ Saved smoothing std plot to: {std_plot}")
        plt.close()

        # Plot best-smoothed speed trace and hist
        best_win = win_labels[np.argmin(std_list)]
        t, s = speed_profiles[best_win]
        duration = phase_time[-1] - phase_time[0]

        plt.figure(figsize=(12, 5))
        plt.plot(t, s, label=f"Speed (win={best_win})", linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (Hz)")
        plt.title("Instantaneous Frequency (Revolution-Based)")
        plt.tight_layout()
        plt.savefig(os.path.splitext(npz_path)[0] + "_smoothed_speed_trace.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        weights = np.ones_like(s) / duration
        plt.hist(s, bins=100, weights=weights, alpha=0.8)
        plt.xlabel("Speed (Hz)")
        plt.ylabel("Normalized Count (/s)")
        plt.title("Speed Distribution")
        plt.tight_layout()
        plt.savefig(os.path.splitext(npz_path)[0] + "_smoothed_speed_hist.png")
        plt.close()

        print(f"✅ Finished smoothed analysis for: {npz_path}")
        print(f"   Best window: {best_win}, std: {min(std_list):.3f} Hz")

    except Exception as e:
        print(f"❌ Failed to analyze {npz_path}: {e}")

if __name__ == "__main__":
    path = "results backup/2025.03.20 patricia/file8/phase_data.npz"  # Update as needed
    analyze_single_file(path)
