import numpy as np
import matplotlib.pyplot as plt
import os


def assign_stator_numbers(step_fitted, threshold):
    """
    Assign discrete stator numbers to step levels using merging threshold,
    then renormalize by quantized spacing.
    """
    levels = []
    indices = []
    n = len(step_fitted)
    current_value = step_fitted[0]
    start = 0

    for i in range(1, n):
        if step_fitted[i] != current_value:
            levels.append(current_value)
            indices.append((start, i))
            current_value = step_fitted[i]
            start = i
    levels.append(current_value)
    indices.append((start, n))

    # Group similar levels
    merged_levels = []
    merged_indices = []
    for i, lvl in enumerate(levels):
        matched = False
        for j, ref_lvl in enumerate(merged_levels):
            if abs(lvl - ref_lvl) <= threshold:
                merged_indices[j].append(indices[i])
                break
        else:
            merged_levels.append(lvl)
            merged_indices.append([indices[i]])

    # Flatten index ranges and recalculate true levels
    flat_indices = []
    final_levels = []
    for group in merged_indices:
        full_range = []
        for start, end in group:
            full_range.extend(range(start, end))
        flat_indices.append(full_range)
        final_levels.append(np.mean(step_fitted[full_range]))

    # Estimate stator step size from level differences
    diffs = np.abs(np.subtract.outer(final_levels, final_levels))
    nonzero_diffs = diffs[(diffs > 1e-3) & (np.triu(np.ones_like(diffs), 1) > 0)]
    delta_v = np.min(nonzero_diffs)
    print(f"Estimated per-stator speed contribution: {delta_v:.2f} Hz")

    # Assign stator numbers based on quantization
    quantized_levels = np.round(np.array(final_levels) / delta_v).astype(int)

    stator_trace = np.zeros_like(step_fitted, dtype=int)
    for group_indices, stator in zip(flat_indices, quantized_levels):
        stator_trace[group_indices] = stator

    return stator_trace, final_levels, flat_indices, quantized_levels


def process_npz(npz_path, threshold):
    data = np.load(npz_path)
    step_fitted = data["step_fitted"]
    t_ref = data["t_ref"]

    stator_trace, merged_levels, segment_indices, stator_nums = assign_stator_numbers(step_fitted, threshold=threshold)

    # Save result
    out_path = os.path.splitext(npz_path)[0] + "_stator.npz"
    np.savez(out_path, t_ref=t_ref, stator_trace=stator_trace, step_fitted=step_fitted, levels=np.array(merged_levels))
    print(f"Saved stator number trace to: {out_path}")

    # Plot for verification
    plt.figure(figsize=(12, 4))
    plt.plot(t_ref, step_fitted, label="Step-Fitted Speed", alpha=0.5)
    for group, stator in zip(segment_indices, stator_nums):
        mid = group[len(group) // 2]
        if mid < len(t_ref):
            plt.text(t_ref[mid], step_fitted[mid], str(stator), ha='center', va='bottom', fontsize=8)
    plt.xlabel("Time (s")
    plt.ylabel("Speed (Hz)")
    plt.title("Step Segments Labeled by Stator Number")
    plt.tight_layout()
    plt.savefig(os.path.splitext(npz_path)[0] + "_stator_assignment_labeled.png")
    plt.close()


if __name__ == "__main__":
    path = "results_backup/2025.04.24 patricia/file/phase_data_speeds.npz"
    process_npz(path, threshold=10.0)
