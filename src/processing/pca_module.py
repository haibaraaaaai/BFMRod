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
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.spatial import KDTree
from numba import njit

from utils.smoothing import smooth_trajectory, smooth_data_with_convolution
from utils.linearizing_function import linearizing_speed_function
from config import (
    SAMPLING_RATE,
    CONVOLUTION_WINDOW,
    REFERENCE_NUM_POINTS,
    WINDOW_DETECTION,
    FIRST_CYCLE_DETECTION_LIMIT,
    END_OF_CYCLE_LIMIT,
    CONTINUITY_CONSTRAINT,
)


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
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(smoothed_data)
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
    index = np.empty(num_points, dtype=np.int32)

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

def ref_cycle_update(X_pca, timestamps, smooth_ref_cycle, start_idx, ref_start):
    """Assign phases, update reference cycles, and return unwrapped phase over time."""
    update_interval_samples = SAMPLING_RATE
    total_samples_pca = X_pca.shape[0]
    updated_refs = [(timestamps[start_idx + ref_start], smooth_ref_cycle)]

    i = 0
    phase0 = np.array([], dtype=np.int32)
    prev_phase = None

    while i < total_samples_pca:
        segment = X_pca[i : min(i + update_interval_samples, total_samples_pca)]
        if segment.shape[0] == 0:
            break

        phase_indices = assign_phase_indices(segment, smooth_ref_cycle, prev_phase)

        # Phase-bin averaging (using last fraction of data)
        phase_bins = [[] for _ in range(len(smooth_ref_cycle))]
        fraction = 0.025
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
                interpolated = 0.5 * (np.mean(phase_bins[left], axis=0) + np.mean(phase_bins[right], axis=0))
                phase_bins[p_idx] = [interpolated]

        # Blend with previous ref cycle
        new_ref = np.array([np.median(points, axis=0) for points in phase_bins])
        ALPHA = 0.2
        blended_ref = (1 - ALPHA) * smooth_ref_cycle + ALPHA * new_ref
        smooth_ref_cycle = smooth_trajectory(blended_ref)
        updated_refs.append((timestamps[start_idx + i], smooth_ref_cycle))

        i += update_interval_samples
        phase0 = np.concatenate((phase0, phase_indices))
        prev_phase = phase_indices[-1]

    phase = phase0 / len(smooth_ref_cycle) * 2 * np.pi
    phase = np.unwrap(phase)

    # Experimental: Chi2 Filtering can smooth phase data while perserving changes.
    # Computation heavy, so swap to Savitzky–Golay filter if that's too slow.
    # phase = chi2_filter_njit_slope_steps(phase, sigma=0.05)

    phase_time = timestamps[start_idx : start_idx + len(phase)]

    return updated_refs, phase, phase_time

# ───────────────────────────────────────────────────────────────
# High-Level Workflow
# ───────────────────────────────────────────────────────────────

def run_pca_workflow(data, timestamps, start_time, end_time, segment_duration, closure_threshold):
    segment_size = int(segment_duration * SAMPLING_RATE)
    total_time = end_time - start_time
    expected_segments = int(total_time / segment_duration)

    pad_samples = CONVOLUTION_WINDOW - 1
    total_samples_needed = expected_segments * segment_size
    raw_end_sample = total_samples_needed + pad_samples

    start_idx = np.searchsorted(timestamps, start_time)
    end_idx = min(start_idx + raw_end_sample, len(timestamps))

    raw_signals = data[start_idx:end_idx, :]
    smoothed_signals = smooth_data_with_convolution(raw_signals)

    X_pca, _ = apply_pca(smoothed_signals)
    # Segment PCA
    pca_segments = []
    for seg_idx in range(expected_segments):
        seg_start = seg_idx * segment_size
        seg_end = seg_start + segment_size
        if seg_end > X_pca.shape[0]:
            break
        segment_data = X_pca[seg_start:seg_end]
        seg_start_time = timestamps[start_idx + pad_samples + seg_start]
        seg_end_time = timestamps[start_idx + pad_samples + seg_end - 1]
        pca_segments.append((segment_data, seg_start_time, seg_end_time))

    # Detect and build initial ref cycle
    ref_start, ref_end = detect_cycle_bounds(X_pca, closure_threshold)
    initial_cycle = X_pca[ref_start:ref_end]
    M = len(initial_cycle)
    avg_signal_d_av = np.zeros([M // REFERENCE_NUM_POINTS, 3])
    for i in range(M // REFERENCE_NUM_POINTS):
        avg_signal_d_av[i] = np.mean(initial_cycle[i * REFERENCE_NUM_POINTS : (i + 1) * REFERENCE_NUM_POINTS], axis=0)
    smooth_ref_cycle = smooth_trajectory(avg_signal_d_av)

    updated_refs, phase, phase_time = ref_cycle_update(X_pca, timestamps, smooth_ref_cycle, start_idx, ref_start)

    phase = savgol_filter(phase, window_length=51, polyorder=3)
    raw_speed = np.gradient(phase, 1 / SAMPLING_RATE)
    phi_wrapped = phase % (2 * np.pi)
    # the plotting within linearizing_function.py seems not to be working, so show=0
    f = linearizing_speed_function(phi_wrapped, raw_speed, N=500, fftfilter=1, mfilter=7, show=0)
    phi_corrected = f(phi_wrapped)
    phase = np.unwrap(phi_corrected)

    return {
        "pca_segments": pca_segments,
        "updated_refs": updated_refs,
        "phase": phase,
        "phase_time": phase_time,
    }
