import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# --- Settings ---
RESULTS_DIR = "results"
GROUP_REVS = 3  # Number of revolutions to average over

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

def plot_speed_traces(all_speeds, all_times, labels):
    plt.figure(figsize=(12, 5))
    for t, s, label in zip(all_times, all_speeds, labels):
        plt.plot(t, s, label=label, linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (Hz)")
    plt.title("Instantaneous Frequency (Revolution-Based)")
    plt.legend()
    plt.tight_layout()

def plot_speed_histograms(all_speeds, labels):
    plt.figure(figsize=(10, 5))
    bins = np.linspace(0, max(max(s) for s in all_speeds), 100)
    for s, label in zip(all_speeds, labels):
        plt.hist(s, bins=bins, alpha=0.5, label=label, histtype='stepfilled')
    plt.xlabel("Speed (Hz)")
    plt.ylabel("Count")
    plt.title("Speed Distribution")
    plt.legend()
    plt.tight_layout()

def find_phase_files(base_dir):
    return sorted(glob.glob(os.path.join(base_dir, "*", "*", "phase_data.npz")))

def main():
    files = find_phase_files(RESULTS_DIR)
    if not files:
        print("⚠ No phase_data.npz files found.")
        return

    all_speeds, all_times, labels = [], [], []

    for path in files:
        try:
            data = np.load(path)
            phase = data["phase"]
            phase_time = data["phase_time"]

            t, s = compute_revolution_frequency(phase, phase_time)
            all_speeds.append(s)
            all_times.append(t)

            folder_label = os.path.join(
                os.path.basename(os.path.dirname(os.path.dirname(path))),
                os.path.basename(os.path.dirname(path))
            )
            labels.append(folder_label)
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")

    plot_speed_traces(all_speeds, all_times, labels)
    plot_speed_histograms(all_speeds, labels)
    plt.show()

if __name__ == "__main__":
    main()
