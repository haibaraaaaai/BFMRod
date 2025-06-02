import pandas as pd
import numpy as np

# === Parameters ===
step_summary_path = "results_backup/2025.05.23 patricia ox/files3/stepfit_speed_trace/step_summary.csv"
change_times = [60, 360, 570, 810]  # in seconds

# === Load and clean step summary ===
df = pd.read_csv(step_summary_path)

# Drop last 2 rows (final fall to 0 + trailing zero row)
df = df.iloc[:-2].copy()

# Compute cumulative step timing
df["Start Time (s)"] = df["Dwell Time (s)"].cumsum().shift(fill_value=0)
df["End Time (s)"] = df["Start Time (s)"] + df["Dwell Time (s)"]
df["Abs Step Size (Hz)"] = df["Step Size (Hz)"].abs()

# Find the closest step to each buffer change time — mark those as buffer-induced (fake)
fake_indices = []
for t_change in change_times:
    closest_idx = np.argmin(np.abs(df["Start Time (s)"] - t_change))
    fake_indices.append(closest_idx)

# Use those fake steps as segment dividers — exclude them from analysis
segment_bounds = [0] + df.loc[fake_indices, "Start Time (s)"].tolist() + [df["End Time (s)"].iloc[-1]]

# Remove the fake steps from the data
df_filtered = df.drop(index=fake_indices).reset_index(drop=True)

# === Segment-wise summary ===
summary = []
for i in range(len(segment_bounds) - 1):
    t_start = segment_bounds[i]
    t_end = segment_bounds[i + 1]
    seg_df = df_filtered[(df_filtered["Start Time (s)"] >= t_start) & (df_filtered["End Time (s)"] <= t_end)]

    summary.append({
        "Segment": i + 1,
        "Start Time (s)": t_start,
        "End Time (s)": t_end,
        "Num Steps": len(seg_df),
        "Mean Dwell Time (s)": seg_df["Dwell Time (s)"].mean() if not seg_df.empty else np.nan,
        "Mean Step Size (Hz)": seg_df["Abs Step Size (Hz)"].mean() if not seg_df.empty else np.nan
    })

summary_df = pd.DataFrame(summary)
print(summary_df)

# Save output
out_path = step_summary_path.replace("step_summary.csv", "step_segments_summary_v2.csv")
summary_df.to_csv(out_path, index=False)
print(f"✅ Segment summary saved to: {out_path}")
