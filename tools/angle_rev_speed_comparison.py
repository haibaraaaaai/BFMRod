import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

# --- Settings ---
REFERENCE_NUM_POINTS = 200
WINDOW_WIDTH_RAD = 0.2 * np.pi
WINDOW_STRIDE_RAD = 0.05 * np.pi
SMOOTH_WINDOW = 51
SMOOTH_POLYORDER = 3
LIMIT_SAMPLES = 500_000
MIN_POINTS = 10
PLOT_SMOOTH_WINDOW = 9
PLOT_SMOOTH_POLYORDER = 2

npz_path = "results_backup/2025.04.15 patricia/files1/phase_data.npz"

# --- Functions ---
def compute_revolution_speed(phase, phase_time):
    """Compute per revolution speed by fitting phase vs time over each full 2π revolution."""
    rev_count = int(phase[-1] // (2 * np.pi))
    speeds = np.full(rev_count, np.nan)
    rev_offsets = np.arange(rev_count) * 2 * np.pi

    for rev in range(rev_count):
        low = rev_offsets[rev]
        high = low + 2 * np.pi
        mask = (phase >= low) & (phase < high)
        if np.count_nonzero(mask) >= 10:  # Require enough points
            x = phase_time[mask]
            y = phase[mask]
            A = np.vstack([x, np.ones_like(x)]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            speed = slope / (2 * np.pi)
            speeds[rev] = speed

    return speeds

def compute_speed_matrix(phase, phase_time, angle_starts, angle_width_rad=0.4, min_points=10):
    n_angles = len(angle_starts)
    rev_count = int(phase[-1] // (2 * np.pi))
    speed_matrix = np.full((n_angles, rev_count), np.nan)

    rev_offsets = np.arange(rev_count) * 2 * np.pi
    angle_bounds = [(start, start + angle_width_rad) for start in angle_starts]

    for rev in range(rev_count):
        rev_offset = rev_offsets[rev]
        for a_idx, (angle_start, angle_end) in enumerate(angle_bounds):
            low = rev_offset + angle_start
            high = rev_offset + angle_end
            mask = (phase >= low) & (phase < high)
            if np.count_nonzero(mask) >= min_points:
                x = phase_time[mask]
                y = phase[mask]
                A = np.vstack([x, np.ones_like(x)]).T
                slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                speed = slope / (2 * np.pi)
                speed_matrix[a_idx, rev] = speed

    return speed_matrix

# --- Main Analysis ---
def main(npz_path):
    data = np.load(npz_path)
    phase0 = data["phase0"][:LIMIT_SAMPLES]
    phase_time = data["phase_time"][:LIMIT_SAMPLES]

    # Phase preprocessing
    phase = phase0 / REFERENCE_NUM_POINTS * 2 * np.pi
    phase = np.unwrap(phase)
    phase = savgol_filter(phase, SMOOTH_WINDOW, SMOOTH_POLYORDER)

    # Speed per rev
    s1 = compute_revolution_speed(phase, phase_time)

    # Speed per angle
    angle_bins = np.arange(0, 2 * np.pi, WINDOW_STRIDE_RAD)
    speed_matrix = compute_speed_matrix(
        phase,
        phase_time,
        angle_bins,
        angle_width_rad=WINDOW_WIDTH_RAD,
        min_points=MIN_POINTS,
    )

    # Align lengths
    min_revs = min(speed_matrix.shape[1], len(s1))
    s1 = s1[:min_revs]
    speed_matrix = speed_matrix[:, :min_revs]

    # Smooth for plotting
    s1_smooth = savgol_filter(s1, PLOT_SMOOTH_WINDOW, PLOT_SMOOTH_POLYORDER)

    # Correlation analysis
    correlations = []
    for i in range(speed_matrix.shape[0]):
        angle_speed = speed_matrix[i]
        mask = ~np.isnan(angle_speed)
        if np.count_nonzero(mask) > 10:
            corr = np.corrcoef(angle_speed[mask], s1[mask])[0,1]
            correlations.append(corr)
        else:
            correlations.append(np.nan)
    correlations = np.array(correlations)

    # --- Plots ---
    base = os.path.splitext(npz_path)[0]

    # Plot example angle speeds vs per-rev speed
    plt.figure(figsize=(12, 6))
    n_angles = speed_matrix.shape[0]
    selected_indices = np.linspace(0, n_angles-1, 8, dtype=int)
    for i in selected_indices:
        angle_trace = speed_matrix[i]
        angle_trace_smooth = savgol_filter(np.nan_to_num(angle_trace, nan=np.nanmean(angle_trace)), PLOT_SMOOTH_WINDOW, PLOT_SMOOTH_POLYORDER)
        plt.plot(angle_trace_smooth, label=f"Angle {i} ({angle_bins[i]/np.pi:.2f}π)", alpha=0.7, linewidth=1)
    plt.plot(s1_smooth, color='black', linestyle='--', linewidth=2, label="Per Rev Speed")
    plt.xlabel("Revolution Index")
    plt.ylabel("Speed (Hz)")
    plt.title("Smoothed Speed vs Revolution for Selected Angles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base + "_speed_angle_vs_rev.png")
    plt.close()

    # Plot normalized (mean-subtracted) traces
    plt.figure(figsize=(12, 6))
    for i in selected_indices:
        angle_trace = speed_matrix[i]
        angle_trace_smooth = savgol_filter(np.nan_to_num(angle_trace, nan=np.nanmean(angle_trace)), PLOT_SMOOTH_WINDOW, PLOT_SMOOTH_POLYORDER)
        normalized_trace = angle_trace_smooth - np.nanmean(angle_trace_smooth)
        plt.plot(normalized_trace, label=f"Angle {i} ({angle_bins[i]/np.pi:.2f}π)", alpha=0.7, linewidth=1)
    normalized_s1 = s1_smooth - np.nanmean(s1_smooth)
    plt.plot(normalized_s1, color='black', linestyle='--', linewidth=2, label="Per Rev Speed (Normalized)")
    plt.xlabel("Revolution Index")
    plt.ylabel("Relative Speed (Hz)")
    plt.title("Normalized Speed Fluctuations vs Revolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base + "_normalized_speed_angle_vs_rev.png")
    plt.close()

    # Plot correlations
    plt.figure(figsize=(8, 5))
    plt.plot(angle_bins/np.pi, correlations, '-o')
    plt.axhline(0, linestyle='--', color='gray')
    plt.xlabel("Angle (π rad)")
    plt.ylabel("Correlation with Per-Rev Speed")
    plt.title("Angle-wise Correlation with Per-Rev Speed")
    plt.ylim(-1,1)
    plt.tight_layout()
    plt.savefig(base + "_angle_correlation.png")
    plt.close()

    print("Done: Plots saved.")

if __name__ == "__main__":
    main(npz_path)
