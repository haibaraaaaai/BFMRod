import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from config import WINDOW_DETECTION, FIRST_CYCLE_DETECTION_LIMIT, END_OF_CYCLE_LIMIT, load_settings

settings = load_settings()
NEIGHBOR_WINDOW_DIVISOR = settings["neighbor_window_divisor"]


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
    return X_pca[:, :3]  # Retain only the first three principal components


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
        variation = np.sum(np.diff(main_component[start: start + END_OF_CYCLE_LIMIT]) ** 2)
        if variation > max_variation:
            max_variation = variation
            best_start = start
    start_index = best_start

    # Compute a median-based threshold for detecting cycle boundaries
    distances_close = np.linalg.norm(
        trajectory[start_index, :] - trajectory[start_index: start_index + WINDOW_DETECTION // NEIGHBOR_WINDOW_DIVISOR, :], axis=1
    )
    close_distance_threshold = np.median(distances_close)

    # Construct a KDTree for efficient nearest-neighbor lookup in PCA space
    tree_trajectory = KDTree(trajectory[start_index: start_index + WINDOW_DETECTION, :])

    # Evaluate neighbor counts for potential cycle end candidates
    nb_neighbours_l = []
    i_list = range(start_index + 400, start_index + END_OF_CYCLE_LIMIT, 10)
    for i in i_list:
        tree_next_points = KDTree(trajectory[i: i + WINDOW_DETECTION, :])
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
