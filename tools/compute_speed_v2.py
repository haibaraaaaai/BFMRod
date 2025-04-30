import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

# --- Settings ---
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
        duration = t_end - t_start
        if duration < 0.001:  # skip too short revs (likely artifact from bad initial phase)
            continue
        freq = 1 / duration

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

        # High smoothing
        phase_raw = np.unwrap(phase0 / REFERENCE_NUM_POINTS * 2 * np.pi)
        phase = savgol_filter(phase_raw, window_length=2001, polyorder=3)

        t_ref, s_ref = compute_revolution_frequency(phase, phase_time)
        plt.figure(figsize=(12, 4))
        plt.plot(t_ref, s_ref, label="Smoothed Speed")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (Hz)")
        plt.title("Smoothed Revolution Speed Trace")
        plt.tight_layout()
        out_path = os.path.splitext(npz_path)[0] + "_speed_trace_smoothed.png"
        plt.savefig(out_path)
        print(f"Saved speed trace plot to: {out_path}")
        plt.close()

    except Exception as e:
        print(f"Failed to analyze {npz_path}: {e}")

if __name__ == "__main__":
    path = "results_backup/2025.02.23 daping/file2/phase_data.npz"
    analyze_single_file(path)
