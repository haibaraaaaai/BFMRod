import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# === Step Fitting Utilities ===

def get_variance(data, inter, pos):
    attempt = inter[:np.searchsorted(inter, pos)] + [pos] + inter[np.searchsorted(inter, pos):]
    delta = list(data[:attempt[0]] - np.mean(data[:attempt[0]]))
    for i in range(len(attempt) - 1):
        seg = data[attempt[i]:attempt[i + 1]]
        if len(seg) > 0:
            delta += list(seg - np.mean(seg))
    delta += list(data[attempt[-1]:] - np.mean(data[attempt[-1]:]))
    return np.mean(np.array(delta) ** 2.)

def get_pos(data, inter, res=50):
    variance = np.inf
    best_pos = None
    for pos in range(res, len(data) - res, res):
        if pos not in inter and 0 < pos < len(data):
            v = get_variance(data, inter, pos)
            if v < variance:
                variance = v
                best_pos = pos
    if best_pos is not None:
        inter = inter[:np.searchsorted(inter, best_pos)] + [best_pos] + inter[np.searchsorted(inter, best_pos):]
    return best_pos, inter, variance

def get_int_av(data, res=5, th=10, limit_ratio=0.99):
    inter = [0]
    variance = 100
    v_random = 200
    real = 10
    max_len = len(data)
    step_count = 0

    print("⏳ Starting step fitting...")

    while variance / v_random < limit_ratio:
        v_random = np.mean([
            get_variance(data, inter, np.random.randint(0, max_len))
            for _ in range(real)
        ])
        best_pos, inter, variance = get_pos(data, inter, res=res)
        step_count += 1

        if best_pos is None:
            print(f"✅ No more valid step positions. Total steps: {step_count}")
            break

        print(f"  ➤ Step {step_count:2d}: added pos={best_pos}, variance={variance:.4f}, ratio={variance / v_random:.4f}")

    inter = sorted(set(inter))
    if inter[-1] != max_len - 1:
        inter.append(max_len - 1)

    step_fitted = np.zeros_like(data)
    for i in range(len(inter) - 1):
        start, end = inter[i], inter[i + 1]
        if end > start:
            avg = np.mean(data[start:end])
            step_fitted[start:end] = avg
        else:
            print(f"⚠️ Skipping invalid segment: start={start}, end={end}")

    print("✅ Step fitting complete.")
    return step_fitted

# === Parameters ===
csv_path = "data/2025.06.10 patricia 80ox/files5/fft_speed_zeroed_C0_raw.csv"
savgol_window = 5  # must be odd
polyorder = 2

# === Load and process CSV ===
df = pd.read_csv(csv_path)
df["Abs Speed"] = np.abs(df["Dominant Frequency (Hz)"])
smoothed_speed = savgol_filter(df["Abs Speed"], window_length=savgol_window, polyorder=polyorder)
step_fitted = get_int_av(smoothed_speed, res=5)

# === Plotting ===
plt.figure(figsize=(10, 5))
plt.plot(df["Time (s)"], df["Abs Speed"], label="Raw |Freq|", alpha=0.3, marker='o', markersize=3)
plt.plot(df["Time (s)"], smoothed_speed, label="Savitzky-Golay", linewidth=2, color='orange')
plt.plot(df["Time (s)"], step_fitted, label="Step-Fitted", linewidth=2, linestyle='--', color='green')
plt.xlabel("Time (s)")
plt.ylabel("Speed (Hz)")
plt.title("Step-Fitted Speed Trace from |Freq|")
plt.grid(True)
plt.legend()
plt.tight_layout()

# === Save output ===
output_dir = os.path.join(os.path.dirname(csv_path), "stepfit_speed_trace")
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "abs_freq_stepfit.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"✅ Step-fitted plot saved: {plot_path}")

df_out = df.copy()
df_out["Smoothed Speed (Hz)"] = smoothed_speed
df_out["Step-Fitted Speed (Hz)"] = step_fitted
df_out.to_csv(os.path.join(output_dir, "abs_freq_stepfit.csv"), index=False)
print("✅ Speed data saved to CSV.")
