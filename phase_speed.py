import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# === Parameters ===
smoothing_window = 1001
polyorder = 2
window_size = 4000

root_dir = "results_checked"

# === Find all npz files recursively
npz_files = list(Path(root_dir).rglob("*.npz"))
print(f"Found {len(npz_files)} .npz files")

for path in npz_files:
    try:
        data = np.load(path)
        phase0 = data["phase0"]
        timestamps = data["phase_time"]

        if len(phase0) < smoothing_window:
            print(f"⚠️ Skipping short file: {path}")
            continue

        # Convert to radians and unwrap
        phase_raw = (phase0 / 200) * 2 * np.pi
        phase_unwrapped = np.unwrap(phase_raw)

        # Apply Savitzky-Golay smoothing
        phase_smooth = savgol_filter(phase_unwrapped, window_length=smoothing_window, polyorder=polyorder)

        # Compute speeds
        speeds = []
        t_centers = []

        for i in range(0, len(phase_smooth) - window_size + 1, window_size):
            start = i
            end = i + window_size

            dphi = phase_smooth[end - 1] - phase_smooth[start]
            dt = timestamps[end - 1] - timestamps[start]
            hz = dphi / (2 * np.pi * dt)
            t_center = 0.5 * (timestamps[start] + timestamps[end - 1])

            speeds.append(hz)
            t_centers.append(t_center)

        # Save CSV
        df = pd.DataFrame({"Time (s)": t_centers, "Speed (Hz)": speeds})
        output_path = path.parent / "smoothed_speed_trace.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error in {path}: {e}")
