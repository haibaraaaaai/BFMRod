import os
import pandas as pd

def extract_steps_from_file(filepath):
    df = pd.read_csv(filepath)
    time = df["Time (s)"].values
    step_speed = df["Step-Fitted Speed (Hz)"].values

    # Find where the step value changes
    changes = [0] + list((step_speed[1:] != step_speed[:-1]).nonzero()[0] + 1)
    changes.append(len(step_speed))

    steps = []
    for i in range(len(changes) - 1):
        start_idx = changes[i]
        end_idx = changes[i + 1]
        dwell_time = time[end_idx - 1] - time[start_idx]
        value = step_speed[start_idx]
        steps.append((i, dwell_time, value))

    # Step size is difference from previous
    step_sizes = [steps[i + 1][2] - steps[i][2] for i in range(len(steps) - 1)]
    step_sizes.append(0)  # No step after last

    return pd.DataFrame({
        "Step Index": [s[0] for s in steps],
        "Dwell Time (s)": [s[1] for s in steps],
        "Step Value (Hz)": [s[2] for s in steps],
        "Step Size (Hz)": step_sizes
    })

def process_folder(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename == "itot_speed_trace_stepfit.csv":
                fullpath = os.path.join(dirpath, filename)
                print(f"Processing: {fullpath}")
                try:
                    step_df = extract_steps_from_file(fullpath)
                    out_csv = os.path.join(dirpath, "step_summary.csv")
                    step_df.to_csv(out_csv, index=False)
                    print(f"Saved summary to: {out_csv}")
                except Exception as e:
                    print(f"❌ Error processing {fullpath}: {e}")

if __name__ == "__main__":
    base_dir = "data/2025.05.23 patricia ox"
    process_folder(base_dir)