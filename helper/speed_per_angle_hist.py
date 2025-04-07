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
N_SEGMENTS = 10

npz_path = "results backup/2025.03.20 patricia/file8/phase_data.npz"


def compute_all_speeds_by_angle(phase, phase_time, candidate_windows):
    all_speeds_by_angle = []

    max_phase = phase[-1]
    for angle_idx, win_start in enumerate(candidate_windows):
        window_speeds = []
        i = 1
        while True:
            low = win_start + i * 2 * np.pi
            high = low + WINDOW_WIDTH_RAD
            if high > max_phase:
                break
            mask = (phase >= low) & (phase < high)
            if np.count_nonzero(mask) < 5:
                i += 1
                continue

            x = phase_time[mask]
            y = phase[mask]
            if len(x) >= 2:
                A = np.vstack([x, np.ones_like(x)]).T
                slope = np.linalg.lstsq(A, y, rcond=None)[0][0]
            speed = slope / (2 * np.pi)
            window_speeds.append(speed)
            i += 1

        all_speeds_by_angle.append(window_speeds)
        print(f"[{angle_idx+1}/{len(candidate_windows)}] angle={win_start/np.pi:.2f}π, samples={len(window_speeds)}")

    return all_speeds_by_angle


def analyze_single_file(npz_path):
    data = np.load(npz_path)
    phase0 = data["phase0"][:LIMIT_SAMPLES]
    phase_time = data["phase_time"][:LIMIT_SAMPLES]
    base = os.path.splitext(npz_path)[0]

    phase = phase0 / REFERENCE_NUM_POINTS * 2 * np.pi
    phase = np.unwrap(phase)
    phase = savgol_filter(phase, SMOOTH_WINDOW, SMOOTH_POLYORDER)

    candidate_windows = np.arange(0, 2 * np.pi, WINDOW_STRIDE_RAD)
    all_speeds_by_angle = compute_all_speeds_by_angle(phase, phase_time, candidate_windows)

    # --- Segment the speeds list into N chunks ---
    dominant_speeds_per_segment = []
    for seg in range(N_SEGMENTS):
        dom_speeds = []
        for speeds in all_speeds_by_angle:
            total_len = len(speeds)
            if total_len < N_SEGMENTS:
                dom_speeds.append(np.nan)
                continue
            start = seg * total_len // N_SEGMENTS
            end = (seg + 1) * total_len // N_SEGMENTS
            seg_speeds = speeds[start:end]
            if len(seg_speeds) == 0:
                dom_speeds.append(np.nan)
                continue
            hist, bin_edges = np.histogram(seg_speeds, bins=100)
            peak_speed = bin_edges[np.argmax(hist)]
            dom_speeds.append(peak_speed)
        dominant_speeds_per_segment.append(dom_speeds)

    # --- Plot dominant speed by angle for all segments ---
    plt.figure(figsize=(10, 5))
    for i, dom_speeds in enumerate(dominant_speeds_per_segment):
        plt.plot(candidate_windows / np.pi, dom_speeds, label=f"Segment {i+1}")
    plt.xlabel("Angle (π rad)")
    plt.ylabel("Dominant Speed (Hz)")
    plt.title("Dominant Speed vs. Angular Offset (10 segments)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base + "_dominant_speed_by_angle_10seg.png")
    plt.close()

    # --- Full-data speed heatmap ---
    speed_bins = np.linspace(0, 200, 150)
    heatmap = np.zeros((len(candidate_windows), len(speed_bins) - 1), dtype=int)
    for i, speeds in enumerate(all_speeds_by_angle):
        if len(speeds) > 0:
            hist, _ = np.histogram(speeds, bins=speed_bins)
            heatmap[i, :] = hist

    plt.figure(figsize=(10, 6))
    extent = [speed_bins[0], speed_bins[-1], 0, 2]
    plt.imshow(heatmap, aspect='auto', extent=extent, origin='lower', cmap='viridis')
    plt.xlabel("Speed (Hz)")
    plt.ylabel("Angle (π rad)")
    plt.yticks(np.linspace(0, 2, 9), [f"{x:.1f}" for x in np.linspace(0, 2, 9)])
    plt.title("Speed Distribution Heatmap by Angular Offset")
    plt.colorbar(label="Counts")
    plt.tight_layout()
    plt.savefig(base + "_speed_angle_heatmap.png")
    plt.close()

    print("✅ All plots saved.")


if __name__ == "__main__":
    analyze_single_file(npz_path)
