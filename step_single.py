import pandas as pd
import numpy as np

# === Parameters ===
csv_path = "results_backup/2025.05.20 patricia ox/files5/stepfit_speed_trace/step_summary.csv"
buffer_switch_times = [420]  # Times when buffer switches from 0 -> 80 mM

# === Load and preprocess ===
df = pd.read_csv(csv_path)
if len(df) > 2:
    df = df.iloc[:-2]  # Drop the last two fake rows
else:
    raise ValueError("Step summary file too short to trim trailing rows.")

# === Convert to numpy for performance ===
times = df["Dwell Time (s)"].values
values = df["Step Value (Hz)"].values
sizes = df["Step Size (Hz)"].values

# === Locate and analyze transitions ===
results = []

for t_switch in buffer_switch_times:
    # Search for closest upward step before or after the switch
    candidate_indices = np.where(sizes > 0)[0]
    if len(candidate_indices) == 0:
        print(f"❌ No upward step found around t={t_switch}s.")
        continue

    # Find index of the upward step closest to t_switch
    total_time = np.cumsum(times)
    distances = np.abs(total_time[candidate_indices] - t_switch)
    closest_up_index = candidate_indices[np.argmin(distances)]

    if sizes[closest_up_index] <= 0:
        print(f"❌ Closest step to t={t_switch}s is not upward.")
        continue

    t_start = np.sum(times[:closest_up_index+1])
    print(f"🔍 Starting from upward step at index {closest_up_index} (t={t_start:.2f}s)")

    for i in range(closest_up_index + 1, len(df)):
        if sizes[i] > 0:
            print(f"⏹️  Stopped at index {i} due to upward step.")
            break
        elif sizes[i] < 0:
            results.append({
                "Buffer Switch Time": t_switch,
                "Start Time (s)": t_start,
                "Dwell Time (s)": times[i],
                "Step Value (Hz)": values[i],
                "Step Size (Hz)": sizes[i]
            })
        t_start += times[i]

# === Output ===
if results:
    result_df = pd.DataFrame(results)
    print("\n✅ Downward steps following each 0->80 mM transition:")
    print(result_df)
else:
    print("❗ No valid downward steps found after any transitions.")
