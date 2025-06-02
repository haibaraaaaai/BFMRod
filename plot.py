import pandas as pd
import matplotlib.pyplot as plt
import os

# === Parameters ===
csv_path = "results_backup/2025.05.23 patricia ox/files3/stepfit_speed_trace/itot_speed_trace_stepfit.csv"
change_times = [60, 360, 570, 810]  # in seconds
labels = ["80 → 0", "0 → 80", "80 → 0", "0 → 80"]

# === Load data ===
df = pd.read_csv(csv_path)
t = df["Time (s)"]
speed = df["Speed (Hz)"]
smoothed = df["Smoothed Speed (Hz)"]
step = df["Step-Fitted Speed (Hz)"]

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(t, speed, label="Raw Speed", alpha=0.3, marker='o', markersize=2)
plt.plot(t, smoothed, label="Savitzky-Golay", color='orange', linewidth=1.5)
plt.plot(t, step, label="Step-Fitted", color='green', linestyle='--', linewidth=2)

# === Add arrows and horizontal annotations ===
y_top = max(speed) * 0.95
arrow_length = (max(speed) - min(speed)) * 0.08
for xc, label in zip(change_times, labels):
    plt.annotate(label,
                 xy=(xc, y_top),
                 xytext=(xc, y_top + arrow_length),
                 ha='center', fontsize=10, color='red',
                 arrowprops=dict(arrowstyle='-|>', color='red', lw=1.5))

plt.xlabel("Time (s)")
plt.ylabel("Speed (Hz)")
plt.title("Step-Fitted Speed Trace with Buffer Change Events")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save
out_path = os.path.splitext(csv_path)[0] + "_with_changes.png"
plt.savefig(out_path, dpi=300)
print(f"✅ Saved updated plot to: {out_path}")
