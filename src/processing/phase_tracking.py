"""
phase_tracking.py – Assign phase indices to PCA trajectory points using nearest-neighbor matching.
"""

import numpy as np
from numba import njit

from config import CONTINUITY_CONSTRAINT


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
