"""
Analyze instantaneous rotation speed as a function of angular position across revolutions.
Generates full and per-segment speed-angle matrices, profiles, and stability plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# Settings
REFERENCE_NUM_POINTS = 200
WINDOW_WIDTH_RAD = 0.2 * np.pi
WINDOW_STRIDE_RAD = 0.05 * np.pi
SMOOTH_WINDOW = 51
SMOOTH_POLYORDER = 3
LIMIT_SAMPLES = 500_000
MIN_POINTS = 10
SEGMENT_SIZE = 250_000  # 1-second chunks at 250 kHz

npz_path = "results_backup/2025.04.15 patricia/file/phase_data.npz"


def compute_speed_matrix(phase, phase_time, angle_starts, angle_width_rad=0.4, min_points=10):
    """Compute [angle_bin, revolution] speed matrix from phase and time."""
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


def analyze_single_file(npz_path):
    # Load data and unwrap phase
    data = np.load(npz_path)
    phase0 = data["phase0"][:LIMIT_SAMPLES]
    phase_time = data["phase_time"][:LIMIT_SAMPLES]
    base = os.path.splitext(npz_path)[0]

    phase = phase0 / REFERENCE_NUM_POINTS * 2 * np.pi
    phase = np.unwrap(phase)
    phase = savgol_filter(phase, SMOOTH_WINDOW, SMOOTH_POLYORDER)

    # Define angle bins
    angle_bins = np.arange(0, 2 * np.pi, WINDOW_STRIDE_RAD)

    # Compute full speed matrix
    speed_matrix = compute_speed_matrix(
        phase,
        phase_time,
        angle_bins,
        angle_width_rad=WINDOW_WIDTH_RAD,
        min_points=MIN_POINTS,
    )

    # Plot full speed matrix
    plt.figure(figsize=(12, 6))
    plt.imshow(speed_matrix, aspect='auto', origin='lower', cmap='viridis',
               extent=[0, speed_matrix.shape[1], 0, 2])
    plt.colorbar(label="Speed (Hz)")
    plt.xlabel("Revolution Index")
    plt.ylabel("Angle (π rad)")
    plt.yticks(np.linspace(0, 2, 9), [f"{x:.1f}" for x in np.linspace(0, 2, 9)])
    plt.title("Speed vs Angle and Revolution")
    plt.tight_layout()
    plt.savefig(base + "_speed_matrix_angle_vs_rev.png")
    plt.close()

    print("Full speed matrix plotted and saved.")

    # Angular profile (mean ± std)
    mean_speed_per_angle = np.nanmean(speed_matrix, axis=1)
    std_speed_per_angle = np.nanstd(speed_matrix, axis=1)

    plt.figure()
    plt.errorbar(angle_bins / np.pi, mean_speed_per_angle, yerr=std_speed_per_angle, fmt='-o', capsize=3)
    plt.xlabel("Angle (π rad)")
    plt.ylabel("Mean Speed (Hz)")
    plt.title("Speed vs Angle (Mean ± Std)")
    plt.tight_layout()
    plt.savefig(base + "_mean_speed_by_angle.png")
    plt.close()

    # Segment-wise analysis
    n_segments = len(phase0) // SEGMENT_SIZE
    segment_profiles = []

    for i in range(n_segments):
        print(f"Segment {i+1}/{n_segments}")
        start = i * SEGMENT_SIZE
        end = (i + 1) * SEGMENT_SIZE

        segment_phase = phase0[start:end] / REFERENCE_NUM_POINTS * 2 * np.pi
        segment_phase = np.unwrap(segment_phase)
        segment_phase = savgol_filter(segment_phase, SMOOTH_WINDOW, SMOOTH_POLYORDER)
        segment_time = phase_time[start:end]

        seg_speed_matrix = compute_speed_matrix(
            segment_phase,
            segment_time,
            angle_bins,
            angle_width_rad=WINDOW_WIDTH_RAD,
            min_points=MIN_POINTS,
        )

        # Plot per-segment speed matrix
        plt.figure(figsize=(10, 4))
        plt.imshow(seg_speed_matrix, aspect='auto', origin='lower', cmap='viridis',
                   extent=[0, seg_speed_matrix.shape[1], 0, 2])
        plt.colorbar(label="Speed (Hz)")
        plt.xlabel("Revolution Index")
        plt.ylabel("Angle (π rad)")
        plt.title(f"Segment {i+1}: Speed vs Angle and Revolution")
        plt.tight_layout()
        plt.savefig(base + f"_segment_{i+1:02d}_speed_matrix.png")
        plt.close()

        # Compute and save mean profile
        mean_per_angle = np.nanmean(seg_speed_matrix, axis=1)
        segment_profiles.append(mean_per_angle)

        plt.figure()
        plt.plot(angle_bins / np.pi, mean_per_angle, '-o')
        plt.xlabel("Angle (π rad)")
        plt.ylabel("Mean Speed (Hz)")
        plt.title(f"Segment {i+1}: Speed vs Angle")
        plt.tight_layout()
        plt.savefig(base + f"_segment_{i+1:02d}_angle_profile.png")
        plt.close()

    # Combine and visualize segment profiles
    segment_profiles = np.array(segment_profiles)  # shape: [n_segments, n_angles]

    # Plot all segment profiles together
    plt.figure()
    for i, profile in enumerate(segment_profiles):
        plt.plot(angle_bins / np.pi, profile, alpha=0.5, label=f"Segment {i+1}" if i < 10 else None)
    plt.xlabel("Angle (π rad)")
    plt.ylabel("Mean Speed (Hz)")
    plt.title("Speed vs Angle for Each 1-Second Segment")
    if n_segments <= 10:
        plt.legend()
    plt.tight_layout()
    plt.savefig(base + "_angle_profiles_per_segment.png")
    plt.close()

    # Plot heatmap of all segment profiles
    plt.figure(figsize=(10, 6))
    plt.imshow(segment_profiles, aspect='auto', cmap='viridis',
               extent=[0, 2, 1, n_segments])
    plt.xlabel("Angle (π rad)")
    plt.ylabel("Segment Index")
    plt.title("Speed Profile Over Time (Heatmap)")
    plt.colorbar(label="Speed (Hz)")
    plt.tight_layout()
    plt.savefig(base + "_segment_profile_heatmap.png")
    plt.close()

    # Plot mean ± std band across segments
    mean_profile = np.nanmean(segment_profiles, axis=0)
    std_profile = np.nanstd(segment_profiles, axis=0)

    plt.figure()
    plt.plot(angle_bins / np.pi, mean_profile, label="Mean")
    plt.fill_between(angle_bins / np.pi,
                     mean_profile - std_profile,
                     mean_profile + std_profile,
                     alpha=0.3, label="±1 std dev")
    plt.xlabel("Angle (π rad)")
    plt.ylabel("Speed (Hz)")
    plt.title("Speed vs Angle (Mean ± Std Dev)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base + "_mean_speed_with_variability_band.png")
    plt.close()

    # Plot variance across segments (Angle Stability)
    angle_variance = np.nanvar(segment_profiles, axis=0)

    plt.figure()
    plt.plot(angle_bins / np.pi, angle_variance, '-o')
    plt.xlabel("Angle (π rad)")
    plt.ylabel("Variance of Mean Speed Across Segments")
    plt.title("Angle Stability Over Time")
    plt.tight_layout()
    plt.savefig(base + "_angle_stability_variance.png")
    plt.close()


if __name__ == "__main__":
    analyze_single_file(npz_path)
