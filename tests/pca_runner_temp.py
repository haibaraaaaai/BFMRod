import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory of the 'processing' module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from processing.pca_module import apply_pca, detect_cycle_bounds
from processing.phase_tracking import assign_phase_indices, update_reference_cycle
from processing.tdms_loader import load_channels_from_tdms
from config.config import START_TIME, END_TIME, REFERENCE_NUM_POINTS
from utils.smoothing import smooth_data_with_convolution, smooth_trajectory

# Reference trajectory (smoothed cycle)
REFERENCE_NUM_POINTS = 200  # Number of points used in the interpolated reference cycle
SMOOTHING = 0.001  # Smoothing factor for B-spline interpolation
DO_UPDATE_REFERENCE_CYCLE = True  # Enable reference cycle updates for drift correction
UPDATE_REFERENCE_CYCLE_SIZE = 250000  # Interval (in points) for updating reference cycle


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Define input TDMS file path
    tdms_file_path = os.path.join(PROJECT_ROOT, "data", "2025.02.23 ox", "file1.tdms")

    # Define results folder
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)  # Ensure results directory exists

    # Construct output file path based on input file
    dir_name, file_name = os.path.split(os.path.relpath(tdms_file_path, os.path.join(PROJECT_ROOT, "data")))
    base_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(RESULTS_DIR, dir_name, f"{base_name}_{START_TIME}_{END_TIME}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"TDMS File Path: {tdms_file_path}")
    print(f"Output Path: {output_path}")

    # Load and preprocess the signals from TDMS file
    C0, C90, C45, C135 = load_channels_from_tdms(tdms_file_path)
    signals = np.column_stack((C0, C90, C45, C135))  # Stack channels into a matrix
    smoothed_signals = smooth_data_with_convolution(signals)  # Apply convolution smoothing

    # Perform PCA transformation
    X_pca, pca_model = apply_pca(smoothed_signals)

    # Load manually set cycle indices if available, otherwise detect automatically
    if os.path.exists(output_path + "_phase_indices_ui.txt"):
        indices = np.loadtxt(output_path + "_phase_indices_ui.txt", dtype=int, delimiter=";")
        print(f"Using manually set cycle bounds: {indices}")
    else:
        indices = detect_cycle_bounds(X_pca)  # Detect start and end indices of the cycle
    np.savetxt(output_path + "_phase_indices.txt", indices, fmt="%i", delimiter=";")  # Save indices

    # Extract and smooth the reference cycle from PCA-transformed data
    avg_signal_d = X_pca[indices[0] : indices[1]]  # Extract one full cycle
    M = len(avg_signal_d)
    avg_signal_d_av = np.zeros([M // REFERENCE_NUM_POINTS, 3])  # Initialize averaged trajectory
    for i in range(0, M // REFERENCE_NUM_POINTS):
        avg_signal_d_av[i] = np.mean(avg_signal_d[i * REFERENCE_NUM_POINTS : (i + 1) * REFERENCE_NUM_POINTS], axis=0)
    smooth_points = smooth_trajectory(avg_signal_d_av)  # Generate smooth reference cycle

    # Initialize phase tracking
    i = 0
    phase0 = np.array([], dtype=np.int32)
    prev_phase = 0

    # Assign phase indices and update reference cycle over time if enabled
    if DO_UPDATE_REFERENCE_CYCLE:
        while i < X_pca.shape[0]:  # Process data in chunks
            # Assign phase indices to trajectory points
            phaseh = assign_phase_indices(
                X_pca[indices[0] + i : indices[0] + i + UPDATE_REFERENCE_CYCLE_SIZE],
                smooth_points,
                prev_phase=prev_phase,
            )
            # Update reference cycle based on assigned phases
            smooth_points = update_reference_cycle(
                phaseh, smooth_points, X_pca[indices[0] + i : indices[0] + i + UPDATE_REFERENCE_CYCLE_SIZE]
            )
            i += UPDATE_REFERENCE_CYCLE_SIZE  # Move to next chunk
            phase0 = np.concatenate((phase0, phaseh))  # Store phase mapping
            prev_phase = phaseh[-1]  # Track last phase for continuity
    else:
        phase0 = assign_phase_indices(X_pca[indices[0] :], smooth_points)  # Assign phase indices without updates

    # Compute phase angles (in radians) and unwrap to maintain continuity
    phase = phase0 / len(smooth_points) * 2 * np.pi
    phase = np.unwrap(phase)

    # --- Plotting Results ---
    fig = plt.figure(figsize=(8, 6), dpi=300)

    # 3D plot of the smoothed reference cycle and latest processed trajectory
    ax = fig.add_subplot(321, projection="3d")
    ax.plot(*smooth_points.T, "-", linewidth=2, label="Smoothed Loop", color="blue")  # Smoothed cycle
    ax.plot(*X_pca[i - UPDATE_REFERENCE_CYCLE_SIZE + indices[0] : i - UPDATE_REFERENCE_CYCLE_SIZE + indices[1]].T, "-", linewidth=2, label="Current Trajectory", color="red")  # Current trajectory

    # 2D comparison of PCA component and its phase-mapped reference
    ax = fig.add_subplot(322)
    ax.plot(X_pca[indices[0] : indices[0] + 20000, 0], color="black")  # Original PCA trajectory
    ax.plot(smooth_points[phase0[:20000], 0])  # Phase-mapped reference trajectory

    # Raw signal data for visual verification
    ax = fig.add_subplot(323)
    ax.plot(signals[indices[0] : indices[0] + 20000])  # Plot raw channel data

    # Phase evolution over time
    ax = fig.add_subplot(324)
    ax.plot(phase, label="Phase")  # Plot computed phase values

    # PCA trajectory and initial cycle detection for verification
    ax = fig.add_subplot(313)
    ax.plot(X_pca[indices[0] : 2 * indices[1] - indices[0], 0], color="black")  # Full trajectory in PCA space
    ax.plot(avg_signal_d[:, 0], color="red")  # Initially detected cycle
    xticks = np.linspace(0, 2 * indices[1] - 2 * indices[0], 40).astype(int)
    xticks = 100 * np.round(xticks / 100).astype(int)  # Adjust for readability
    ax.set_xticks(xticks)
    labels = [str(indices[0] + xt) for xt in xticks]  # Label indices
    ax.set_xticklabels(labels, rotation=90)
    for tick in ax.get_xticks():
        ax.axvline(x=tick, color="gray", linestyle="--", linewidth=0.5)  # Add vertical guide lines

    fig.tight_layout()
    fig.savefig(output_path + "_phase_debug.png")  # Save the figure

    # --- Save Output Data ---
    np.save(output_path + "_phase.npy", phase)  # Save computed phase data
    ## np.save(output_path + "_smooth_points.npy", smooth_points[phase0, 0])  # Save phase-mapped trajectory
    ## np.save(output_path + "_avg_signal.npy", X_pca[indices[0] :, 0])  # Save PCA trajectory
