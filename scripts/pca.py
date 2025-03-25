import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, peak_widths
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from nptdms import TdmsFile
from numba import njit

# Sampling and Time Settings
SAMPLING_RATE = 250000  # Hz (samples per second)
START_TIME = 0  # Start time of data segment (seconds)
END_TIME = 3  # End time of data segment (seconds)

# Pre-processing
CONVOLUTION_WINDOW = 100  # Window size for signal smoothing via moving average

# Cycle detection parameters
WINDOW_DETECTION = 400  # Sliding window size for periodicity detection
FIRST_CYCLE_DETECTION_LIMIT = 500000  # Search for cycle start within first N points
END_OF_CYCLE_LIMIT = 100000  # Search for cycle end within this range after start
NEIGHBOR_WINDOW_DIVSOR = 100  # Determines resolution for local distance comparison

# Reference trajectory (smoothed cycle)
REFERENCE_NUM_POINTS = 200  # Number of points used in the interpolated reference cycle
SMOOTHING = 0.001  # Smoothing factor for B-spline interpolation
DO_UPDATE_REFERENCE_CYCLE = True  # Enable reference cycle updates for drift correction
UPDATE_REFERENCE_CYCLE_SIZE = 250000  # Interval (in points) for updating reference cycle

# Phase assignment settings
CONTINUITY_CONSTRAINT = REFERENCE_NUM_POINTS // 10  # Search neighborhood size for phase continuity

def load_channels_from_tdms(tdms_file_path):
    """
    Load the four primary channel signals from a TDMS file within the specified time window.

    Args:
        tdms_file_path (str): Path to the TDMS file

    Returns:
        tuple: 4 NumPy arrays corresponding to signals C0, C90, C45, and C135.
    """
    tdms_file = TdmsFile.read(tdms_file_path)
    group = tdms_file.groups()[0]
    channels = group.channels()

    # Extract signal data within the defined time range
    C90 = channels[0].data[START_TIME * SAMPLING_RATE : END_TIME * SAMPLING_RATE]
    C45 = channels[1].data[START_TIME * SAMPLING_RATE : END_TIME * SAMPLING_RATE]
    C135 = channels[2].data[START_TIME * SAMPLING_RATE : END_TIME * SAMPLING_RATE]
    C0 = channels[3].data[START_TIME * SAMPLING_RATE : END_TIME * SAMPLING_RATE]

    return C0, C90, C45, C135


def smooth_data_with_convolution(signals, window_size=CONVOLUTION_WINDOW):
    """
    Apply a moving average filter to smooth the input signals.

    Args:
        signals (np.ndarray): Raw multi-channel signal data
        window_size (int): Size of the moving average window

    Returns:
        np.ndarray: Smoothed signal data, same shape as input (except for window-induced reduction).
    """
    smoothed_signals = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(window_size) / window_size, mode="valid"),
        axis=0,
        arr=signals,
    )
    return smoothed_signals


def apply_pca(smoothed_data, n_components=4):
    """
    Perform Principal Component Analysis (PCA) to reduce dimensionality of smoothed signals.

    Args:
        smoothed_data (np.ndarray): Preprocessed signal data
        n_components (int): Number of PCA components to retain

    Returns:
        tuple:
            - X_pca (np.ndarray): Transformed PCA components (first three principal components)
            - pca (PCA): PCA model containing eigenvectors and explained variance
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(smoothed_data)
    X_pca = X_pca[:, :3]  # Retain only the first three principal components
    return X_pca, pca


def detect_cycle_bounds(trajectory):
    """
    Identify the start and end indices of a representative cycle in the PCA trajectory.

    Args:
        trajectory (np.ndarray): PCA-transformed data (3D trajectory)

    Returns:
        tuple: (start_index, end_index) of the detected cycle
    """
    # Extract the first PCA component (assumed to capture the dominant cycle behavior)
    main_component = trajectory[:FIRST_CYCLE_DETECTION_LIMIT, 0]  
    minp = np.min(main_component)
    maxp = np.max(main_component)
    amplitude_90 = minp + 0.9 * (maxp - minp)  # Define threshold at 90% of max amplitude

    # Identify candidate start indices where the main component crosses the threshold
    possible_starts = np.where(main_component > amplitude_90)[0]

    # Filter candidate starts to ensure sufficient separation between detected points
    min_distance = 1000  # Minimum separation between detected starts
    filtered_starts = [possible_starts[0]]
    for idx in possible_starts[1:]:
        if idx - filtered_starts[-1] > min_distance:
            filtered_starts.append(idx)
        if len(filtered_starts) > 5:  # Limit to 5 potential cycle starts
            break
    possible_starts = np.array(filtered_starts)

    # Select the best start index based on maximum variation within the assumed cycle
    max_variation = 0
    best_start = possible_starts[0]
    for start in possible_starts:
        variation = np.sum(np.diff(main_component[start : start + END_OF_CYCLE_LIMIT]) ** 2)
        if variation > max_variation:
            max_variation = variation
            best_start = start
    start_index = best_start

    # Compute a median-based threshold for detecting cycle boundaries
    distances_close = np.linalg.norm(
        trajectory[start_index, :] - trajectory[start_index : start_index + WINDOW_DETECTION // NEIGHBOR_WINDOW_DIVSOR, :], axis=1
    )
    close_distance_threshold = np.median(distances_close)

    # Construct a KDTree for efficient nearest-neighbor lookup in PCA space
    tree_trajectory = KDTree(trajectory[start_index : start_index + WINDOW_DETECTION, :])

    # Evaluate neighbor counts for potential cycle end candidates
    nb_neighbours_l = []
    i_list = range(start_index + 400, start_index + END_OF_CYCLE_LIMIT, 10)
    for i in i_list:
        tree_next_points = KDTree(trajectory[i : i + WINDOW_DETECTION, :])
        nb_neighbours = tree_next_points.count_neighbors(tree_trajectory, close_distance_threshold)
        nb_neighbours_l.append(nb_neighbours)

    # Identify peaks in neighbor count, indicating periodic overlap of cycles
    threshold = 2  # Initial prominence threshold
    peaks = []
    while len(peaks) == 0:  # Adjust threshold until a peak is found
        peaks = find_peaks(nb_neighbours_l, prominence=WINDOW_DETECTION / threshold)[0]
        threshold *= 2

    # Measure peak widths to determine where cycles most likely end
    widths, _, _, right_ips = peak_widths(nb_neighbours_l, peaks, rel_height=0.5)
    right_ips = right_ips.astype(int)

    end_index = peaks[0]
    peak_ind = 0

    # Refine peak selection to minimize false positives in cycle detection
    while peak_ind < len(peaks) - 1:
        if np.min(nb_neighbours_l[: peaks[peak_ind]]) > 2 * np.min(nb_neighbours_l):
            peak_ind += 1
        else:
            break

    # Further refine peak selection based on local maxima
    for k in range(peak_ind, min(peak_ind + 3, len(peaks))):
        if nb_neighbours_l[peaks[k]] > nb_neighbours_l[peaks[peak_ind]] * 1.5:
            peak_ind = k

    # Define end of the cycle as the right boundary of the selected peak
    end_index = i_list[right_ips[peak_ind]]

    return start_index, end_index


def smooth_trajectory(points):
    """
    Apply cubic B-spline interpolation to smooth a closed 3D trajectory.

    Args:
        points (np.ndarray): Array of 3D points defining the trajectory

    Returns:
        np.ndarray: Smoothed trajectory points with consistent spacing
    """
    points = np.array(points)

    # Ensure the trajectory is a closed loop (repeat first point at the end)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    # Fit a parametric 3D spline with periodic boundary conditions
    tck, _ = splprep(points.T, per=True, s=SMOOTHING)
    u_fine = np.linspace(0, 1, REFERENCE_NUM_POINTS)  # Interpolation parameter
    smooth_points = np.array(splev(u_fine, tck)).T

    return smooth_points


@njit
def assign_phase_indices(trajectory, reference_cycle, prev_phase=None):
    """
    Assigns phase indices to trajectory points by mapping them to the closest points 
    on the reference cycle using nearest-neighbor matching.

    Args:
        trajectory (np.ndarray): The PCA-transformed trajectory data (N x 3).
        reference_cycle (np.ndarray): The reference cycle representing a single oscillation (M x 3).
        prev_phase (int, optional): The last known phase index for continuity correction.

    Returns:
        np.ndarray: Array of phase indices indicating the closest match for each trajectory point.
    """
    num_points = trajectory.shape[0]  # Number of points in the trajectory segment
    len_array = reference_cycle.shape[0]  # Number of points in the reference cycle
    index = np.empty(num_points, dtype=np.int32)  # Initialize output array

    # Step 1: Assign phase for the first point
    if prev_phase is not None:
        # If previous phase is given, restrict search to nearby reference points for continuity
        neighboring_indices = (prev_phase - np.arange(-CONTINUITY_CONSTRAINT, CONTINUITY_CONSTRAINT)) % len_array

        # Compute Euclidean distances to nearby reference cycle points
        diff = reference_cycle[neighboring_indices] - trajectory[0, :3]
        distances = np.sum(diff**2, axis=1)  # Squared distance (no need for sqrt)
        
        # Assign closest phase index
        best_distance = np.argmin(distances)
        index[0] = neighboring_indices[best_distance]
    else:
        # If no previous phase, search across the entire reference cycle
        diff = reference_cycle - trajectory[0, :3]
        distances = np.sum(diff**2, axis=1)
        best_distance = np.argmin(distances)
        index[0] = best_distance

    # Step 2: Assign phase indices for remaining trajectory points
    for i in range(1, num_points):
        last_index = index[i - 1]

        # Restrict search to neighboring points in reference cycle for continuity
        neighboring_indices = (last_index - np.arange(-CONTINUITY_CONSTRAINT, CONTINUITY_CONSTRAINT)) % len_array

        # Compute Euclidean distances to nearby reference cycle points
        diff = reference_cycle[neighboring_indices] - trajectory[i, :3]
        distances = np.sum(diff**2, axis=1)

        # Assign closest phase index
        best_distance = np.argmin(distances)
        index[i] = neighboring_indices[best_distance]

    return index


def update_reference_cycle(phase_indices, reference_cycle, trajectory):
    """
    Updates the reference cycle using assigned phases by averaging corresponding 
    trajectory points mapped to each phase index.

    Args:
        phase_indices (np.ndarray): Phase indices mapping trajectory points to reference cycle points.
        reference_cycle (np.ndarray): Current reference cycle representing a full oscillation.
        trajectory (np.ndarray): Input trajectory data for updating the reference cycle.

    Returns:
        np.ndarray: Updated reference cycle after averaging corresponding points.
    """
    new_traj = np.zeros_like(reference_cycle)  # Initialize updated reference cycle

    for i in range(reference_cycle.shape[0]):  # Iterate over each reference cycle index
        exp_points = np.where(phase_indices == i)[0]  # Find trajectory points mapped to this phase index

        if len(exp_points) == 0:
            # If no trajectory points were mapped to this phase, retain the previous reference value
            new_traj[i] = reference_cycle[i]
        else:
            # Average the last 100 mapped points for a stable update
            new_traj[i] = np.mean(trajectory[exp_points[-100:]], axis=0)

    # Smooth the updated reference cycle to maintain continuity
    new_traj = smooth_trajectory(new_traj)

    return new_traj


if __name__ == "__main__":
    # Define input TDMS file path
    tdms_file_path = "data/2025.02.23 ox/file1.tdms"

    # Construct output directory structure based on input file path
    dir_name, file_name = os.path.split(os.path.relpath(tdms_file_path, "data"))
    base_name = os.path.splitext(file_name)[0]
    output_path = os.path.join("results", dir_name, f"{base_name}_{START_TIME}_{END_TIME}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directory exists

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
