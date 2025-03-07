import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, peak_widths
from scipy.spatial import KDTree
from nptdms import TdmsFile

CONVOLUTION_WINDOW = 100  
WINDOW_DETECTION = 400  
FIRST_CYCLE_DETECTION_LIMIT = 500000  
END_OF_CYCLE_LIMIT = 100000  

def load_channels_from_tdms(tdms_file_path):
    if not os.path.isfile(tdms_file_path):
        raise FileNotFoundError(f"File {tdms_file_path} does not exist.")

    try:
        tdms_file = TdmsFile.read(tdms_file_path)
        group = tdms_file.groups()[0]
        channels = group.channels()

        if len(channels) < 4:
            raise ValueError(f"Expected at least 4 channels, but found {len(channels)}.")

        C90 = channels[0].data[:250000]
        C45 = channels[1].data[:250000]
        C135 = channels[2].data[:250000]
        C0 = channels[3].data[:250000]

        return C0, C90, C45, C135

    except Exception as e:
        raise ValueError(f"Error loading TDMS file {tdms_file_path}: {e}")


def smooth_data_with_convolution(signals, window_size=CONVOLUTION_WINDOW):
    smoothed_signals = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(window_size) / window_size, mode="valid"),
        axis=0,
        arr=signals,
    )

    return smoothed_signals

def apply_pca(smoothed_data, n_components=4):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(smoothed_data)
    X_pca = X_pca[:, :3]

    return X_pca, pca


def detect_cycle_bounds(trajectory, signals):
    main_component = trajectory[:FIRST_CYCLE_DETECTION_LIMIT, 0]  
    minp = np.min(main_component)
    maxp = np.max(main_component)
    amplitude_90 = minp + 0.9 * (maxp - minp)

    possible_starts = np.where(main_component > amplitude_90)[0]

    min_distance = 1000  
    filtered_starts = [possible_starts[0]]
    for idx in possible_starts[1:]:
        if idx - filtered_starts[-1] > min_distance:
            filtered_starts.append(idx)
        if len(filtered_starts) > 5:
            break
    possible_starts = np.array(filtered_starts)

    max_variation = 0
    best_start = possible_starts[0]
    for start in possible_starts:
        variation = np.sum(np.diff(main_component[start : start + END_OF_CYCLE_LIMIT]) ** 2)
        if variation > max_variation:
            max_variation = variation
            best_start = start
    start_index = best_start

    distances_close = np.linalg.norm(
        signals[start_index, :] - signals[start_index : start_index + 2, :], axis=1
    )
    close_distance_threshold = np.max(distances_close)

    tree_trajectory = KDTree(
        trajectory[start_index : start_index + WINDOW_DETECTION, :]
    )

    nb_neighbours_l = []
    i_list = range(start_index + 400, start_index + END_OF_CYCLE_LIMIT, 10)

    for i in i_list:
        tree_next_points = KDTree(trajectory[i : i + WINDOW_DETECTION, :])
        nb_neighbours = tree_next_points.count_neighbors(
            tree_trajectory, close_distance_threshold
        )
        nb_neighbours_l.append(nb_neighbours)

    threshold = 2
    peaks = []
    while len(peaks) == 0:
        peaks = find_peaks(nb_neighbours_l, prominence=WINDOW_DETECTION / threshold)[0]
        threshold *= 2

    widths, _, _, right_ips = peak_widths(nb_neighbours_l, peaks, rel_height=0.5)
    right_ips = right_ips.astype(int)

    print(peaks)#
    print(threshold)#

    end_index = peaks[0]
    peak_ind = 0

    while peak_ind < len(peaks) - 1:
        if np.min(nb_neighbours_l[: peaks[peak_ind]]) > 2 * np.min(nb_neighbours_l):
            peak_ind += 1
        else:
            break

    for k in range(peak_ind, min(peak_ind + 3, len(peaks))):
        if nb_neighbours_l[peaks[k]] > nb_neighbours_l[peaks[peak_ind]] * 1.5:
            peak_ind = k
    if len(right_ips) > peak_ind:
        end_index = i_list[right_ips[peak_ind]]
    else:
        raise ValueError(f"Invalid peak index {peak_ind} for right_ips of size {len(right_ips)}")

    return start_index, end_index


def plot_pca_cycle(X_pca, start_index, end_index):
    """
    Plots PCA trajectory and highlights the detected cycle.

    Parameters:
    - X_pca (np.ndarray): PCA-transformed data.
    - start_index (int): Start index of detected cycle.
    - end_index (int): End index of detected cycle.
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], color="gray", alpha=0.3)
    ax.plot(X_pca[start_index:end_index, 0], 
            X_pca[start_index:end_index, 1], 
            X_pca[start_index:end_index, 2], 
            color="red", linewidth=2)

    ax.set_title("Detected Cycle in PCA Space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.show()


if __name__ == "__main__":
    tdms_file_path = "data/2025.02.23 ox/file.tdms"
    output_path = tdms_file_path[:-5]

    try:
        C0, C90, C45, C135 = load_channels_from_tdms(tdms_file_path)
        signals = np.column_stack((C0, C90, C45, C135))
        indices = None
        smoothed_signals = smooth_data_with_convolution(signals)

        X_pca, pca_model = apply_pca(smoothed_signals)

        if os.path.exists(output_path + "_phase_indices_ui.txt"):
            indices = np.loadtxt(output_path + "_phase_indices_ui.txt", dtype=int, delimiter=";")
            print(f"Using manually set cycle bounds: {indices}")
        else:
            indices = detect_cycle_bounds(X_pca, smoothed_signals)
        np.savetxt(output_path + "_phase_indices.txt", indices, fmt="%i", delimiter=";")

                # Plot detected cycle
        plot_pca_cycle(X_pca, indices[0], indices[1])

    except Exception as e:
        print(f"Error processing {tdms_file_path}: {e}")