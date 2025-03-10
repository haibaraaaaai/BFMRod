import numpy as np
from processing.pca_module import apply_pca, detect_cycle_bounds
from processing.phase_tracking import assign_phase_indices, update_reference_cycle
from processing.tdms_loader import load_tdms_data
from config import REFERENCE_NUM_POINTS, load_settings
from utils import smooth_data_with_convolution, smooth_trajectory

settings = load_settings()
START_TIME = settings["start_time"]
END_TIME = settings["end_time"]
DO_UPDATE_REFERENCE_CYCLE = settings.get("do_update_reference_cycle", True)
UPDATE_REFERENCE_CYCLE_SIZE = settings.get("update_reference_cycle_size", 250000)


def run_pca_analysis(tdms_file_path):
    """
    Runs PCA processing and phase assignment for a given TDMS file.

    Args:
        tdms_file_path (str): Path to the TDMS data file.

    Returns:
        dict: Processed PCA data, phase indices, and reference cycle.
    """
    # Load and preprocess signals
    signals = load_tdms_data(tdms_file_path)
    smoothed_signals = smooth_data_with_convolution(signals)

    # Perform PCA
    X_pca = apply_pca(smoothed_signals)

    # Detect cycle bounds
    indices = detect_cycle_bounds(X_pca)

    # Extract and smooth reference cycle
    avg_signal_d = X_pca[indices[0]: indices[1]]
    avg_signal_d_av = np.array([
        np.mean(avg_signal_d[i * REFERENCE_NUM_POINTS: (i + 1) * REFERENCE_NUM_POINTS], axis=0)
        for i in range(len(avg_signal_d) // REFERENCE_NUM_POINTS)
    ])
    smooth_points = smooth_trajectory(avg_signal_d_av)

    # Assign phase indices
    i = 0
    phase0 = np.array([], dtype=np.int32)
    prev_phase = 0

    if DO_UPDATE_REFERENCE_CYCLE:
        while i < X_pca.shape[0]:
            phaseh = assign_phase_indices(
                X_pca[indices[0] + i: indices[0] + i + UPDATE_REFERENCE_CYCLE_SIZE],
                smooth_points,
                prev_phase=prev_phase,
            )
            smooth_points = update_reference_cycle(
                phaseh, smooth_points, X_pca[indices[0] + i: indices[0] + i + UPDATE_REFERENCE_CYCLE_SIZE]
            )
            i += UPDATE_REFERENCE_CYCLE_SIZE
            phase0 = np.concatenate((phase0, phaseh))
            prev_phase = phaseh[-1]
    else:
        phase0 = assign_phase_indices(X_pca[indices[0]:], smooth_points)

    # Compute phase angles
    phase = phase0 / len(smooth_points) * 2 * np.pi
    phase = np.unwrap(phase)

    # Return processed data
    return {
        "X_pca": X_pca,
        "phase": phase,
        "smooth_points": smooth_points,
        "indices": indices,
    }
