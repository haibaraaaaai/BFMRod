"""
pca_module.py – Centralized PCA and phase tracking workflow for TDMS signal data.
Handles:
    - Signal smoothing
    - PCA transformation
    - Reference cycle detection and updating
    - Phase assignment
    - Full analysis workflow callable from GUI or scripts
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, peak_widths
from scipy.spatial import KDTree
from numba import njit

from utils.smoothing import smooth_trajectory
from config import (
    SAMPLING_RATE,
    WINDOW_DETECTION,
    FIRST_CYCLE_DETECTION_LIMIT,
    END_OF_CYCLE_LIMIT,
    CONTINUITY_CONSTRAINT,
)
from processing.normalize_channels import normalize_signals


# ───────────────────────────────────────────────────────────────
# Core PCA and Phase Functions
# ───────────────────────────────────────────────────────────────

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
    smoothed_data_normalized = normalize_signals(smoothed_data, method="percentile", percentile=95)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(smoothed_data_normalized)
    return X_pca[:, :3], pca

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
    right_ips = np.round(right_ips).astype(int)

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

@njit
def assign_phase_indices(trajectory, reference_cycle, prev_phase=None):
    """
    Assign phase indices to trajectory points by mapping them to the closest points
    on the reference cycle using nearest-neighbor matching.

    Args:
        trajectory (np.ndarray): PCA-transformed trajectory data (N x 3).
        reference_cycle (np.ndarray): Reference cycle representing a full oscillation (M x 3).
        prev_phase (int, optional): Last known phase index for continuity correction.

    Returns:
        np.ndarray: Array of phase indices indicating closest match for each trajectory point.
    """
    num_points = trajectory.shape[0]
    len_array = reference_cycle.shape[0]
    index = np.empty(num_points, dtype=np.uint8)

    # Assign phase for first point
    if prev_phase is not None:
        neighboring_indices = (prev_phase - np.arange(-CONTINUITY_CONSTRAINT, CONTINUITY_CONSTRAINT)) % len_array
        diff = reference_cycle[neighboring_indices] - trajectory[0, :3]
        distances = np.sum(diff**2, axis=1)
        best_distance = np.argmin(distances)
        index[0] = neighboring_indices[best_distance]
    else:
        diff = reference_cycle - trajectory[0, :3]
        distances = np.sum(diff**2, axis=1)
        index[0] = np.argmin(distances)

    # Assign phases for remaining points
    for i in range(1, num_points):
        last_index = index[i - 1]
        neighboring_indices = (last_index - np.arange(-CONTINUITY_CONSTRAINT, CONTINUITY_CONSTRAINT)) % len_array
        diff = reference_cycle[neighboring_indices] - trajectory[i, :3]
        distances = np.sum(diff**2, axis=1)
        best_distance = np.argmin(distances)
        index[i] = neighboring_indices[best_distance]

    return index

def ref_cycle_update(X_pca, computed_ref, update_interval, fraction=0.025, alpha=0.2):
    """Assign phases, update reference cycles from computed_refs, return unwrapped phase over time."""
    update_sample_size = int(update_interval * SAMPLING_RATE)
    total_samples_pca = X_pca.shape[0]
    if total_samples_pca == 0:
        return [], np.array([], dtype=np.uint8)

    updated_refs = []
    phase0 = np.array([], dtype=np.uint8)
    prev_phase = None

    ref_start_idx, smooth_ref_cycle = computed_ref
    end_idx = total_samples_pca

    i = 0
    while i < end_idx:
        seg_start = i
        seg_end = min(i + update_sample_size, end_idx)
        segment = X_pca[seg_start : seg_end]
        if segment.shape[0] == 0:
            break

        phase_indices = assign_phase_indices(segment, smooth_ref_cycle, prev_phase)

        if i + update_sample_size < end_idx:
            phase_bins = [[] for _ in range(len(smooth_ref_cycle))]
            cut_start = int(len(segment) * (1 - fraction))
            for idx in range(cut_start, len(segment)):
                phase_bins[phase_indices[idx]].append(segment[idx])

            # Interpolate missing phases
            for p_idx in range(len(phase_bins)):
                if not phase_bins[p_idx]:
                    left = (p_idx - 1) % len(smooth_ref_cycle)
                    right = (p_idx + 1) % len(smooth_ref_cycle)
                    while not phase_bins[left]:
                        left = (left - 1) % len(smooth_ref_cycle)
                    while not phase_bins[right]:
                        right = (right + 1) % len(smooth_ref_cycle)
                    interpolated = 0.5 * (
                        np.mean(phase_bins[left], axis=0) + np.mean(phase_bins[right], axis=0)
                    )
                    phase_bins[p_idx] = [interpolated]

            # Blend with previous ref cycle
            new_ref = np.array([np.median(points, axis=0) for points in phase_bins])
            blended_ref = (1 - alpha) * smooth_ref_cycle + alpha * new_ref
            smooth_ref_cycle = smooth_trajectory(blended_ref)
            updated_refs.append((ref_start_idx + seg_end, smooth_ref_cycle))

        i += update_sample_size

        phase0 = np.concatenate((phase0, phase_indices))
        prev_phase = phase_indices[-1]

    return updated_refs, phase0
