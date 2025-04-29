import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import imageio
import os
from collections import defaultdict

# --- Settings ---
REFERENCE_NUM_POINTS = 200
WINDOW_STRIDE_RAD = 0.04 * np.pi
SMOOTH_WINDOW = 51
SMOOTH_POLYORDER = 3
LIMIT_SAMPLES = 5_000_000
MIN_POINTS = 2

npz_path = "results_backup/2025.04.15 patricia/files/phase_data.npz"

def compute_speed_matrix_fast(phase, phase_time, phase_width=0.02*np.pi, min_points=2):
    n_bins = int(np.ceil(phase[-1] / phase_width))
    rev_count = int(np.floor(phase[-1] / (2*np.pi)))
    bins_per_rev = int(np.round((2*np.pi) / phase_width))

    speed_matrix = np.full((bins_per_rev, rev_count), np.nan)

    bin_edges = np.arange(0, (n_bins+1)*phase_width, phase_width)
    bin_indices = np.digitize(phase, bin_edges) - 1

    # Precompute bin to indices mapping
    bin_to_indices = defaultdict(list)
    for i, b in enumerate(bin_indices):
        bin_to_indices[b].append(i)

    for rev in range(rev_count):
        if rev % 100 == 0:
            print(f"Fitting revolution {rev}/{rev_count}")
        start_bin = int(rev * (2*np.pi) / phase_width)
        for bin_offset in range(bins_per_rev):
            bin_idx = start_bin + bin_offset
            indices = bin_to_indices.get(bin_idx, [])
            if len(indices) >= min_points:
                x = phase_time[indices]
                y = phase[indices]
                A = np.vstack([x, np.ones_like(x)]).T
                slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                speed = slope / (2*np.pi)
                speed_matrix[bin_offset, rev] = speed

    return speed_matrix

def main(npz_path):
    data = np.load(npz_path)
    phase0 = data["phase0"][:LIMIT_SAMPLES]
    phase_time = data["phase_time"][:LIMIT_SAMPLES]

    output_dir = os.path.dirname(npz_path)
    output_gif_path = os.path.join(output_dir, "polar_speed_evolution.gif")

    # Phase preprocessing
    phase = phase0 / REFERENCE_NUM_POINTS * 2 * np.pi
    phase = np.unwrap(phase)
    phase = savgol_filter(phase, SMOOTH_WINDOW, SMOOTH_POLYORDER)

    # Speed per angle
    angle_bins = np.arange(0, 2 * np.pi, WINDOW_STRIDE_RAD)
    speed_matrix = compute_speed_matrix_fast(
        phase,
        phase_time,
        phase_width=WINDOW_STRIDE_RAD,
        min_points=MIN_POINTS,
    )

    # Align dimensions
    _, n_revs = speed_matrix.shape

    # Prepare frames for gif
    frames = []
    for rev in range(n_revs):
        if rev % 100 == 0:
            print(f"Processed revolution {rev}/{n_revs}")
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, polar=True)
        ax.set_ylim(0, np.nanmax(speed_matrix) * 1.1)  # radial limit

        angles = angle_bins
        speeds = speed_matrix[:, rev]

        # Close the loop
        angles = np.append(angles, angles[0])
        speeds = np.append(speeds, speeds[0])

        ax.plot(angles, speeds, marker='o')
        ax.set_title(f"Revolution {rev}", va='bottom')
        
        # Save frame to memory
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame[:, :, :3])  # Drop alpha channel
        plt.close(fig)

    # Save to gif
    imageio.mimsave(output_gif_path, frames, fps=10)  # 10 frames per second
    print(f"Saved gif to {output_gif_path}")

    # Plot average speed per angle
    mean_speeds = np.nanmean(speed_matrix, axis=1)
    angles = angle_bins
    angles = np.append(angles, angles[0])
    mean_speeds = np.append(mean_speeds, mean_speeds[0])

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_ylim(0, np.nanmax(mean_speeds) * 1.1)
    ax.plot(angles, mean_speeds, marker='o')
    ax.set_title("Average Speed per Angle", va='bottom')
    plt.tight_layout()

    mean_plot_path = os.path.join(output_dir, "polar_speed_mean.png")
    plt.savefig(mean_plot_path)
    plt.close(fig)
    print(f"Saved mean polar plot to {mean_plot_path}")

if __name__ == "__main__":
    main(npz_path)