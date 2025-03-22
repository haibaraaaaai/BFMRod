import numpy as np
from numba import njit
from config import CONTINUITY_CONSTRAINT
# from utils import smooth_trajectory
# from scipy.interpolate import splprep, splev


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

# def update_reference_cycle(phase_indices, reference_cycle, trajectory):
#     """
#     Updates the reference cycle using assigned phases by averaging corresponding 
#     trajectory points mapped to each phase index.

#     Args:
#         phase_indices (np.ndarray): Phase indices mapping trajectory points to reference cycle points.
#         reference_cycle (np.ndarray): Current reference cycle representing a full oscillation.
#         trajectory (np.ndarray): Input trajectory data for updating the reference cycle.

#     Returns:
#         np.ndarray: Updated reference cycle after averaging corresponding points.
#     """
#     new_traj = np.zeros_like(reference_cycle)  # Initialize updated reference cycle

#     for i in range(reference_cycle.shape[0]):  # Iterate over each reference cycle index
#         exp_points = np.where(phase_indices == i)[0]  # Find trajectory points mapped to this phase index

#         if len(exp_points) == 0:
#             # If no trajectory points were mapped to this phase, retain the previous reference value
#             new_traj[i] = reference_cycle[i]
#         else:
#             # Average the last 100 mapped points for a stable update
#             new_traj[i] = np.mean(trajectory[exp_points[-100:]], axis=0)

#     # Smooth the updated reference cycle to maintain continuity
#     new_traj = smooth_trajectory(new_traj)

#     return new_traj
