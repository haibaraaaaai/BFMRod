import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import traceback

"""
Analyze instantaneous speed data from phase outputs across multiple recordings.
Generates speed traces and histograms for early and full durations.
"""

# --- Settings ---
BASE_DIR = "results_backup/2025.04.15 patricia"
GROUP_REVS = 1
PLOT_DURATION = 60  # seconds
HIST_BINS = 100
REFERENCE_NUM_POINTS = 200


def compute_revolution_frequency(phase, phase_time, group_revs=GROUP_REVS):
    """Compute frequency over time using grouped phase revolution intervals."""
    step = 2 * np.pi
    max_phase = phase[-1]
    thresholds = np.arange(step, max_phase, step)

    rev_times = []
    for threshold in thresholds:
        idx = np.searchsorted(phase, threshold)
        if idx >= len(phase_time):
            break
        rev_times.append(phase_time[idx])

    rev_times = np.array(rev_times)  # No phase_time[0] padding

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


def analyze_all_files(base_dir):
    """Process all NPZ files in base_dir and generate speed plots/histograms."""
    npz_files = sorted(glob.glob(os.path.join(base_dir, "**", "phase_data*.npz"), recursive=True))

    if not npz_files:
        print("No phase_data.npz files found.")
        return

    speeds_60s = []
    speeds_full = []
    times_60s = []
    labels = []

    for path in npz_files:
        try:
            data = np.load(path)
            phase0 = data["phase0"]
            phase_time = data["phase_time"]

            # Convert to float phase and unwrap
            phase = phase0 / 200 * 2 * np.pi
            phase = np.unwrap(phase)
            if "harmonic2x" in path.lower():
                phase *= 0.5

            label = os.path.join(
                os.path.basename(os.path.dirname(os.path.dirname(path))),
                os.path.basename(os.path.dirname(path))
            )

            # Full time analysis
            t_full, s_full = compute_revolution_frequency(phase, phase_time)
            speeds_full.append(s_full)

            # First 60s analysis
            duration = phase_time[-1] - phase_time[0]
            max_time = min(PLOT_DURATION, duration)
            idx_60 = np.where(phase_time - phase_time[0] <= max_time)[0]

            if len(idx_60) > 1:
                p60, pt60 = phase[idx_60], phase_time[idx_60]
                t60, s60 = compute_revolution_frequency(p60, pt60)
                speeds_60s.append(s60)
                times_60s.append(t60)
            else:
                speeds_60s.append(np.array([]))
                times_60s.append(np.array([]))

            labels.append(label)
            print(f"{label}: {len(s60)} speeds in first {max_time:.1f}s, {len(s_full)} total speeds")

        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            traceback.print_exc()

    if not speeds_60s or not speeds_full:
        print("No valid speed data to plot.")
        return

    all_combined = np.concatenate(speeds_full)
    max_speed = np.max(all_combined)
    bins = np.linspace(0, max_speed, HIST_BINS)

    # Speed trace from first 60s
    plt.figure(figsize=(12, 5))
    for t, s, label in zip(times_60s, speeds_60s, labels):
        if len(t) > 0:
            plt.plot(t, s, label=label, linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (Hz)")
    plt.title(f"Instantaneous Speed (first {PLOT_DURATION}s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "speed_trace_60s.png"))
    plt.close()
    print("Saved: speed_trace_60s.png")

    # Histogram from first 60s
    plt.figure(figsize=(10, 5))
    for s, label in zip(speeds_60s, labels):
        if len(s) > 0:
            plt.hist(s, bins=bins, alpha=0.4, label=label, histtype="stepfilled")
    plt.xlabel("Speed (Hz)")
    plt.ylabel("Count")
    plt.title(f"Speed Histogram (first {PLOT_DURATION}s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "histogram_60s.png"))
    plt.close()
    print("Saved: histogram_60s.png")

    # Histogram from full data
    plt.figure(figsize=(10, 5))
    for s, label in zip(speeds_full, labels):
        if len(s) > 0:
            plt.hist(s, bins=bins, alpha=0.4, label=label, histtype="stepfilled")
    plt.xlabel("Speed (Hz)")
    plt.ylabel("Count")
    plt.title("Speed Histogram – Full")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "histogram_full.png"))
    plt.close()
    print("Saved: histogram_full.png")

    # Combined histogram (full)
    plt.figure(figsize=(8, 5))
    plt.hist(all_combined, bins=bins, alpha=0.7, color="black")
    plt.xlabel("Speed (Hz)")
    plt.ylabel("Count")
    plt.title("Combined Speed Histogram – Full")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "histogram_combined.png"))
    plt.close()
    print("Saved: histogram_combined.png")

    # Combined histogram (first 60s only)
    all_combined_60s = np.concatenate([s for s in speeds_60s if len(s) > 0])
    plt.figure(figsize=(8, 5))
    plt.hist(all_combined_60s, bins=bins, alpha=0.7, color="gray")
    plt.xlabel("Speed (Hz)")
    plt.ylabel("Count")
    plt.title("Combined Speed Histogram – First 60s")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "histogram_combined_60s.png"))
    plt.close()
    print("Saved: histogram_combined_60s.png")


if __name__ == "__main__":
    print("Analyzing speeds...")
    analyze_all_files(BASE_DIR)
