import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import os

# --- Settings ---
REFERENCE_NUM_POINTS = 200
WINDOW_WIDTH_RAD = 0.2 * np.pi
WINDOW_STRIDE_RAD = 0.05 * np.pi
SMOOTH_WINDOW = 51
SMOOTH_POLYORDER = 3
LIMIT_SAMPLES = 2_500_000  # Analyze only first 250k samples (~1s)

# --- File path (adjust if needed) ---
npz_path = "results backup/2025.03.20 patricia/file8/phase_data.npz"


def detect_revolutions(phase_unwrapped):
    thresholds = np.arange(2 * np.pi, phase_unwrapped[-1], 2 * np.pi)
    rev_starts = [0] + [np.searchsorted(phase_unwrapped, th) for th in thresholds]
    return rev_starts


def get_best_window_phase_speed(phase, phase_time):
    phase_unwrapped = np.unwrap(phase)
    phase_unwrapped = savgol_filter(phase_unwrapped, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER)

    rev_starts = detect_revolutions(phase_unwrapped)
    candidate_windows = np.arange(0, 2 * np.pi, WINDOW_STRIDE_RAD)

    all_speeds_by_angle = []
    all_rmse_by_angle = []

    for win_start in candidate_windows:
        window_speeds = []
        window_rmses = []

        for i in range(1, len(rev_starts) - 1):  # skip first rev
            start, end = rev_starts[i], rev_starts[i + 1]
            p = phase_unwrapped[start:end]
            t = phase_time[start:end]

            if len(p) < 10:
                continue

            local_phase = p % (2 * np.pi)
            mask = (local_phase >= win_start) & (local_phase < win_start + WINDOW_WIDTH_RAD)
            if np.count_nonzero(mask) < 5:
                continue

            x = t[mask].reshape(-1, 1)
            y = p[mask]
            model = LinearRegression().fit(x, y)
            slope = model.coef_[0] / (2 * np.pi)  # Hz
            pred = model.predict(x)
            rmse = np.sqrt(np.mean((y - pred) ** 2))

            window_speeds.append(slope)
            window_rmses.append(rmse)

        all_speeds_by_angle.append(window_speeds)
        all_rmse_by_angle.append(np.mean(window_rmses) if window_rmses else np.inf)

    best_idx = np.argmin(all_rmse_by_angle)
    best_angle = candidate_windows[best_idx]
    return (
        all_speeds_by_angle[best_idx],
        candidate_windows,
        all_speeds_by_angle,
        best_angle,
    )


def analyze_single_file(npz_path):
    data = np.load(npz_path)
    phase0 = data["phase0"][:LIMIT_SAMPLES]
    phase_time = data["phase_time"][:LIMIT_SAMPLES]
    phase = phase0 / REFERENCE_NUM_POINTS * 2 * np.pi

    best_speeds, window_angles, all_speeds_by_angle, best_angle = get_best_window_phase_speed(phase, phase_time)
    base = os.path.splitext(npz_path)[0]

    # --- Plot best speed trace ---
    plt.figure(figsize=(12, 5))
    plt.plot(best_speeds, marker="o")
    plt.title(f"Best Speed per Rev (angle offset = {best_angle / np.pi:.2f}π)")
    plt.xlabel("Revolution #")
    plt.ylabel("Speed (Hz)")
    plt.tight_layout()
    plt.savefig(base + "_DEBUG_speed_best_window.png")
    plt.close()

    # --- Plot speed traces from all angular windows ---
    plt.figure(figsize=(12, 6))
    for angle, speeds in zip(window_angles, all_speeds_by_angle):
        if len(speeds) > 0:
            plt.plot(speeds, label=f"{angle / np.pi:.2f}π", alpha=0.5)
    plt.title("Speed per Revolution by Angular Window (first 1s)")
    plt.xlabel("Revolution #")
    plt.ylabel("Speed (Hz)")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(base + "_DEBUG_speed_by_angle_windows.png")
    plt.close()

    print(f"✅ Debug plots saved for: {os.path.basename(npz_path)}")
    print(f"   Best angle offset: {best_angle:.2f} rad")


if __name__ == "__main__":
    analyze_single_file(npz_path)
