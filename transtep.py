import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# === Power-law model ===
def power_law(x, a, b):
    return a * np.power(x, b)

# === Configuration ===
file_timer_pairs = [
    ("results_backup/2025.05.23 patricia ox/files3/stepfit_speed_trace/step_summary.csv", [360, 810]),
    ("results_backup/2025.05.23 patricia ox/files1/stepfit_speed_trace/step_summary.csv", [643]),
    ("results_backup/2025.05.23 patricia ox/files5/stepfit_speed_trace/step_summary.csv", [450, 910, 1230]),
    ("results_backup/2025.05.23 patricia ox/files6/stepfit_speed_trace/step_summary.csv", [360]),
    ("results_backup/2025.05.20 patricia ox/file1/stepfit_speed_trace/step_summary.csv", [430]),
    ("results_backup/2025.05.20 patricia ox/files3/stepfit_speed_trace/step_summary.csv", [360]),
    ("results_backup/2025.05.20 patricia ox/files5/stepfit_speed_trace/step_summary.csv", [420])
]

all_results = []

for idx, (csv_path, buffer_switch_times) in enumerate(file_timer_pairs):
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    if len(df) < 3:
        print(f"⚠️  Too few rows in {csv_path}, skipping.")
        continue

    df = df.iloc[:-2]  # Remove last 2 rows (fake terminal steps)
    times = df["Dwell Time (s)"].values
    values = df["Step Value (Hz)"].values
    sizes = df["Step Size (Hz)"].values
    total_time = np.cumsum(times)

    for t_switch in buffer_switch_times:
        candidate_indices = np.where(sizes > 0)[0]
        if len(candidate_indices) == 0:
            print(f"❌ No upward step found around t={t_switch}s in {csv_path}")
            continue

        distances = np.abs(total_time[candidate_indices] - t_switch)
        closest_up_index = candidate_indices[np.argmin(distances)]

        if sizes[closest_up_index] <= 0:
            print(f"❌ Closest step to t={t_switch}s is not upward in {csv_path}")
            continue

        file_label = f"file{idx+1}"
        t_start = np.sum(times[:closest_up_index+1])
        found = False

        for i in range(closest_up_index + 1, len(df)):
            if sizes[i] > 0:
                break
            elif sizes[i] < 0:
                found = True
                all_results.append({
                    "File": file_label,
                    "Buffer Switch Time": t_switch,
                    "Start Time (s)": t_start,
                    "Dwell Time (s)": times[i],
                    "Motor Speed (Hz)": values[i],
                    "Step Size (Hz)": abs(sizes[i])
                })
            t_start += times[i]

        if not found:
            print(f"⚠️  No downward steps found after t={t_switch}s in {csv_path}")

result_df = pd.DataFrame(all_results)

# === Fit models ===
x = result_df["Motor Speed (Hz)"].values
y = result_df["Dwell Time (s)"].values
x_fit = x[y > 0]
y_fit = y[y > 0]

# Power-law fit
params, _ = curve_fit(power_law, x_fit, y_fit)
a, b = params
x_line = np.linspace(min(x_fit), max(x_fit), 500)
y_power = power_law(x_line, a, b)

# Log-log linear fit
log_x = np.log(x_fit)
log_y = np.log(y_fit)
log_slope, log_intercept = np.polyfit(log_x, log_y, 1)
y_loglog = np.exp(log_intercept) * x_line ** log_slope

# === Plotting ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=result_df, x="Motor Speed (Hz)", y="Dwell Time (s)",
                hue="File", style="File", s=100, edgecolor="black", linewidth=0.5)

plt.plot(x_line, y_power, color='black', linewidth=2, label=f"Power-Law Fit\ny = {a:.2f}·x^{b:.2f}")
plt.plot(x_line, y_loglog, color='gray', linestyle='--', linewidth=2, label=f"Log-Log Fit\ny = {np.exp(log_intercept):.2f}·x^{log_slope:.2f}")

plt.title("Dwell Time vs Motor Speed\nPower-Law and Log-Log Fits", fontsize=14)
plt.xlabel("Motor Speed (Hz)", fontsize=12)
plt.ylabel("Dwell Time (s)", fontsize=12)
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
