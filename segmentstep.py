import pandas as pd
import numpy as np
import os

# === User Inputs ===
baseline_csv = "results_backup/2025.05.23 patricia ox/files2/stepfit_speed_trace/step_summary.csv"
exchange_csv = "results_backup/2025.05.23 patricia ox/files3/stepfit_speed_trace/step_summary.csv"
change_times = [60, 360, 570, 810]  # buffer change times in seconds

# === Load & Preprocess ===
def load_step_summary(csv_path):
    df = pd.read_csv(csv_path)
    df = df.iloc[:-2].copy()  # remove last two entries (fake step to 0)
    df["Start Time (s)"] = df["Dwell Time (s)"].cumsum().shift(fill_value=0)
    df["End Time (s)"] = df["Start Time (s)"] + df["Dwell Time (s)"]
    df["Abs Step Size (Hz)"] = df["Step Size (Hz)"].abs()
    return df

baseline_df = load_step_summary(baseline_csv)
exchange_df = load_step_summary(exchange_csv)

# === Find real step transitions closest to buffer change times ===
def get_segment_bounds(df, change_times):
    bounds = [0]
    exclude_indices = []
    for t in change_times:
        idx = np.argmin(np.abs(df["Start Time (s)"] - t))
        exclude_indices.append(idx)  # exclude this transition dwell
        bounds.append(df["Start Time (s)"].iloc[idx + 1])  # next step begins next segment
    bounds.append(df["End Time (s)"].iloc[-1])
    return bounds, exclude_indices

exchange_bounds, exclude_idxs = get_segment_bounds(exchange_df, change_times)
baseline_bounds = [0, baseline_df["End Time (s)"].iloc[-1]]

# === Compute dwell/step statistics for each segment ===
def compute_segment_stats(df, bounds, exclude_indices=[]):
    df = df.drop(index=exclude_indices) if exclude_indices else df
    stats = []
    for i in range(len(bounds) - 1):
        seg = df[(df["Start Time (s)"] >= bounds[i]) & (df["End Time (s)"] <= bounds[i + 1])]
        stats.append({
            "Segment": i + 1,
            "Start Time (s)": bounds[i],
            "End Time (s)": bounds[i + 1],
            "Num Steps": len(seg),
            "Mean Dwell Time (s)": seg["Dwell Time (s)"].mean() if not seg.empty else np.nan,
            "Mean Step Size (Hz)": seg["Abs Step Size (Hz)"].mean() if not seg.empty else np.nan
        })
    return pd.DataFrame(stats)

# === Output Comparison Table ===
baseline_stats = compute_segment_stats(baseline_df, baseline_bounds)
baseline_stats.insert(0, "Trace", "Baseline 80mM")

exchange_stats = compute_segment_stats(exchange_df, exchange_bounds, exclude_idxs)
exchange_labels = ["Exchange 80mM" if i % 2 == 0 else "Exchange 0mM" for i in range(len(exchange_stats))]
exchange_stats.insert(0, "Trace", exchange_labels)

summary = pd.concat([baseline_stats, exchange_stats], ignore_index=True)
print(summary)

# === Optional: Save output ===
summary.to_csv("segment_comparison_summary.csv", index=False)