import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

root_dir = "data/2025.05.23 patricia ox"
step_sizes = []
file_labels = []

# We'll build step_sizes as before, but x_vals will be generated below with integer indices
current_file_index = 0
x_vals = []
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith("step_summary.csv"):
            file_path = os.path.join(dirpath, filename)
            try:
                df = pd.read_csv(file_path)
                print(f"✅ Read {file_path}, {len(df)} rows")
                if "Step Size (Hz)" not in df.columns or len(df) < 3:
                    print(f"⚠️ Skipped {file_path} (missing column or too short)")
                    continue  # skip files that are too short
                steps = df["Step Size (Hz)"][:-2].abs().values
                print(f"  ➤ {len(steps)} valid step sizes extracted")
                step_sizes.extend(steps)
                # Instead of file_labels, build x_vals directly with integer indices
                x_vals.extend([current_file_index + 1] * len(steps))
                current_file_index += 1
            except Exception as e:
                print(f"❌ Failed to read {file_path}: {e}")

if not step_sizes:
    print("⚠️ No valid step sizes found. Check your folder and file contents.")
else:
    # Plotting
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=x_vals, y=step_sizes, scale='width', inner='point')
    plt.xlabel("File Index")
    plt.ylabel("Step Size (Hz)")
    plt.title("Distribution of Step Sizes (Violin Plot)")
    plt.tight_layout()
    plt.savefig("step_size_violin.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=x_vals, y=step_sizes)
    plt.xlabel("File Index")
    plt.ylabel("Step Size (Hz)")
    plt.title("Step Size Distribution Across Files (Box Plot)")
    plt.tight_layout()
    plt.savefig("step_size_boxplot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.stripplot(x=x_vals, y=step_sizes, jitter=True, alpha=0.5)
    plt.xlabel("File Index")
    plt.ylabel("Step Size (Hz)")
    plt.title("Step Sizes with Jitter for Visibility")
    plt.tight_layout()
    plt.savefig("step_size_jittered.png", dpi=300)
    plt.close()