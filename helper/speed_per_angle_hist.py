import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# --- Settings ---
REFERENCE_NUM_POINTS = 200
WINDOW_WIDTH_RAD = 0.2 * np.pi
WINDOW_STRIDE_RAD = 0.05 * np.pi
SMOOTH_WINDOW = 51
SMOOTH_POLYORDER = 3
LIMIT_SAMPLES = 250_000
MIN_POINTS = 10  # Minimum points per angle-window to attempt speed estimation

npz_path = "results backup/2025.03.20 patricia/file8/phase_data.npz"


def compute_speed_matrix(phase, phase_time, angle_starts, angle_width_rad=0.4, min_points=10):
    """
    Compute a 2D matrix of speeds: [angle_bin_index, revolution_index]
    """
    n_angles = len(angle_starts)
    rev_count = int(phase[-1] // (2 * np.pi))
    speed_matrix = np.full((n_angles, rev_count), np.nan)

    for rev in range(rev_count):
        for a_idx, angle_start in enumerate(angle_starts):
            low = rev * 2 * np.pi + angle_start
            high = low + angle_width_rad
            mask = (phase >= low) & (phase < high)
            if np.count_nonzero(mask) >= min_points:
                x = phase_time[mask]
                y = phase[mask]
                slope, _ = np.polyfit(x, y, 1)
                speed = slope / (2 * np.pi)
                speed_matrix[a_idx, rev] = speed

    return speed_matrix


def analyze_single_file(npz_path):
    # --- Load and unwrap phase ---
    data = np.load(npz_path)
    phase0 = data["phase0"][:LIMIT_SAMPLES]
    phase_time = data["phase_time"][:LIMIT_SAMPLES]
    base = os.path.splitext(npz_path)[0]

    phase = phase0 / REFERENCE_NUM_POINTS * 2 * np.pi
    phase = np.unwrap(phase)
    phase = savgol_filter(phase, SMOOTH_WINDOW, SMOOTH_POLYORDER)

    # --- Define angle bins ---
    angle_bins = np.arange(0, 2 * np.pi, WINDOW_STRIDE_RAD)

    # --- Compute speed matrix ---
    speed_matrix = compute_speed_matrix(
        phase,
        phase_time,
        angle_bins,
        angle_width_rad=WINDOW_WIDTH_RAD,
        min_points=MIN_POINTS,
    )

    # --- Plot speed matrix heatmap ---
    plt.figure(figsize=(12, 6))
    plt.imshow(speed_matrix, aspect='auto', origin='lower', cmap='viridis',
               extent=[0, speed_matrix.shape[1], 0, 2])
    plt.colorbar(label="Speed (Hz)")
    plt.xlabel("Revolution Index")
    plt.ylabel("Angle (π rad)")
    plt.yticks(np.linspace(0, 2, 9), [f"{x:.1f}" for x in np.linspace(0, 2, 9)])
    plt.title("Speed by Angle and Revolution")
    plt.tight_layout()
    plt.savefig(base + "_speed_matrix_angle_vs_rev.png")
    plt.close()

    print("✅ Speed matrix plotted and saved.")


if __name__ == "__main__":
    analyze_single_file(npz_path)
