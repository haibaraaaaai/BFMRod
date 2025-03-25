"""
pca_module.py – PCA transformation and representative cycle detection on 3D trajectories.
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

from config import (
    WINDOW_DETECTION,
    FIRST_CYCLE_DETECTION_LIMIT,
    END_OF_CYCLE_LIMIT,
)


def apply_pca(smoothed_data, n_components=4):
    """
    Perform Principal Component Analysis (PCA) to reduce signal dimensionality.

    Args:
        smoothed_data (np.ndarray): Preprocessed signal data.
        n_components (int): Number of PCA components to compute.

    Returns:
        tuple:
            - X_pca (np.ndarray): Transformed PCA data (N x 3).
            - pca (PCA): Trained PCA model.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(smoothed_data)
    return X_pca[:, :3], pca  # Return first 3 components + model


def detect_cycle_bounds(trajectory, closure_threshold=40):
    """
    Identify start and end indices of a representative cycle in PCA-transformed trajectory.

    Args:
        trajectory (np.ndarray): PCA-transformed data (N x 3).
        closure_threshold (int): Window size to assess closure proximity (default 40).

    Returns:
        tuple:
            - start_index (int): Index of cycle start.
            - end_index (int): Index of cycle end.
    """
    # Step 1: Use first PCA component to identify candidate cycle starts
    main_component = trajectory[:FIRST_CYCLE_DETECTION_LIMIT, 0]
    amplitude_90 = np.min(main_component) + 0.9 * (np.max(main_component) - np.min(main_component))
    possible_starts = np.where(main_component > amplitude_90)[0]

    # Filter starts to ensure sufficient separation
    min_distance = 1_000
    filtered_starts = [possible_starts[0]]
    for idx in possible_starts[1:]:
        if idx - filtered_starts[-1] > min_distance:
            filtered_starts.append(idx)
        if len(filtered_starts) > 5:
            break
    possible_starts = np.array(filtered_starts)

    # Select best start based on max variation
    best_start, max_variation = possible_starts[0], 0
    for start in possible_starts:
        variation = np.sum(np.diff(main_component[start:start + END_OF_CYCLE_LIMIT]) ** 2)
        if variation > max_variation:
            max_variation = variation
            best_start = start
    start_index = best_start

    # Step 2: Detect cycle end via neighbor overlap analysis
    distances_close = np.linalg.norm(
        trajectory[start_index] - trajectory[start_index:start_index + closure_threshold],
        axis=1
    )
    close_distance_threshold = np.median(distances_close)

    # Create KDTree for reference segment
    ref_tree = KDTree(trajectory[start_index:start_index + WINDOW_DETECTION])

    # Compare neighbor counts for sliding windows
    i_list = range(start_index + 400, start_index + END_OF_CYCLE_LIMIT, 10)
    neighbor_counts = []
    for i in i_list:
        next_tree = KDTree(trajectory[i:i + WINDOW_DETECTION])
        count = next_tree.count_neighbors(ref_tree, close_distance_threshold)
        neighbor_counts.append(count)

    # Find peaks in neighbor counts (cycle overlaps)
    threshold = 2
    peaks = []
    while len(peaks) == 0:
        peaks = find_peaks(neighbor_counts, prominence=WINDOW_DETECTION / threshold)[0]
        threshold *= 2

    _, _, _, right_ips = peak_widths(neighbor_counts, peaks, rel_height=0.5)
    right_ips = right_ips.astype(int)

    # Refine peak selection
    peak_ind = 0
    while peak_ind < len(peaks) - 1:
        if np.min(neighbor_counts[:peaks[peak_ind]]) > 2 * np.min(neighbor_counts):
            peak_ind += 1
        else:
            break
    for k in range(peak_ind, min(peak_ind + 3, len(peaks))):
        if neighbor_counts[peaks[k]] > 1.5 * neighbor_counts[peaks[peak_ind]]:
            peak_ind = k

    end_index = i_list[right_ips[peak_ind]]
    return start_index, end_index
