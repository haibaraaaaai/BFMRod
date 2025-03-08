import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, peak_widths
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from nptdms import TdmsFile
from numba import njit

SAMPLING_RATE = 250000
NUMBER_OF_DATA_POINTS = 250000 * 30 * 60
CONVOLUTION_WINDOW = 100
WINDOW_DETECTION = 400
FIRST_CYCLE_DETECTION_LIMIT = 500000
END_OF_CYCLE_LIMIT = 100000
NEIGHBOR_WINDOW_DIVSOR = 10
REFERENCE_NUM_POINTS = 200
SMOOTHING = 0.001
DO_UPDATE_REFERENCE_CYCLE = True
UPDATE_REFERENCE_CYCLE_SIZE = 250000
CONTINUITY_CONSTRAINT = REFERENCE_NUM_POINTS // 10

def load_channels_from_tdms(tdms_file_path):
    tdms_file = TdmsFile.read(tdms_file_path)
    group = tdms_file.groups()[0]
    channels = group.channels()

    C90 = channels[0].data[:NUMBER_OF_DATA_POINTS]
    C45 = channels[1].data[:NUMBER_OF_DATA_POINTS]
    C135 = channels[2].data[:NUMBER_OF_DATA_POINTS]
    C0 = channels[3].data[:NUMBER_OF_DATA_POINTS]

    return C0, C90, C45, C135


def smooth_data_with_convolution(signals, window_size=CONVOLUTION_WINDOW):
    smoothed_signals = np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size) / window_size, mode="valid"), axis=0, arr=signals)
    return smoothed_signals


def apply_pca(smoothed_data, n_components=4):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(smoothed_data)
    X_pca = X_pca[:, :3]
    return X_pca, pca


def detect_cycle_bounds(trajectory):
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

    distances_close = np.linalg.norm(trajectory[start_index, :] - trajectory[start_index : start_index + WINDOW_DETECTION // NEIGHBOR_WINDOW_DIVSOR, :], axis=1)
    close_distance_threshold = np.median(distances_close)

    tree_trajectory = KDTree(trajectory[start_index : start_index + WINDOW_DETECTION, :])
    nb_neighbours_l = []
    i_list = range(start_index + 400, start_index + END_OF_CYCLE_LIMIT, 10)

    for i in i_list:
        tree_next_points = KDTree(trajectory[i : i + WINDOW_DETECTION, :])
        nb_neighbours = tree_next_points.count_neighbors(tree_trajectory, close_distance_threshold)
        nb_neighbours_l.append(nb_neighbours)

    threshold = 2
    peaks = []
    while len(peaks) == 0:
        peaks = find_peaks(nb_neighbours_l, prominence=WINDOW_DETECTION / threshold)[0]
        threshold *= 2

    widths, _, _, right_ips = peak_widths(nb_neighbours_l, peaks, rel_height=0.5)
    right_ips = right_ips.astype(int)

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

    end_index = i_list[right_ips[peak_ind]]

    return start_index, end_index


def smooth_trajectory(points):
    points = np.array(points)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    tck, _ = splprep(points.T, per=True, s=SMOOTHING)
    u_fine = np.linspace(0, 1, REFERENCE_NUM_POINTS)
    smooth_points = np.array(splev(u_fine, tck)).T
    return smooth_points


@njit
def assign_phase_indices(trajectory, reference_cycle, prev_phase=None):
    index = np.empty(len(trajectory), dtype=np.int32)
    len_array = reference_cycle.shape[0]
    num_points = trajectory.shape[0]

    if prev_phase is not None:
        neighboring_indices = (prev_phase - np.arange(-CONTINUITY_CONSTRAINT, CONTINUITY_CONSTRAINT)) % len_array

        diff = reference_cycle[neighboring_indices] - trajectory[0, :3]
        distancesar = np.sum(diff**2, axis=1)

        best_distance = np.argmin(distancesar)
        index[0] = neighboring_indices[best_distance]
    else:
        diff = reference_cycle - trajectory[0, :3]
        distancesar = np.sum(diff**2, axis=1)
        best_distance = np.argmin(distancesar)
        index[0] = best_distance

    for i in range(1, num_points):
        last_index = index[i - 1]
        neighboring_indices = (last_index - np.arange(-CONTINUITY_CONSTRAINT, CONTINUITY_CONSTRAINT)) % len_array
        diff = reference_cycle[neighboring_indices] - trajectory[i, :3]
        distancesar = np.sum(diff**2, axis=1)

        best_distance = np.argmin(distancesar)
        index[i] = neighboring_indices[best_distance]

    return index


def update_reference_cycle(phase_indices, reference_cycle, trajectory):
    new_traj = np.zeros_like(reference_cycle)
    for i in range(reference_cycle.shape[0]):
        exp_points = np.where(phase_indices == i)[0]
        if len(exp_points) == 0:
            new_traj[i] = reference_cycle[i]
        else:
            new_traj[i] = np.mean(trajectory[exp_points[-100:]], axis=0)
    new_traj = smooth_trajectory(new_traj)
    return new_traj


if __name__ == "__main__":
    tdms_file_path = "data/2025.02.23 ox/file1.tdms"
    output_path = tdms_file_path[:-5]

    C0, C90, C45, C135 = load_channels_from_tdms(tdms_file_path)
    signals = np.column_stack((C0, C90, C45, C135))
    indices = None
    smoothed_signals = smooth_data_with_convolution(signals)

    X_pca, pca_model = apply_pca(smoothed_signals)

    if os.path.exists(output_path + "_phase_indices_ui.txt"):
        indices = np.loadtxt(output_path + "_phase_indices_ui.txt", dtype=int, delimiter=";")
        print(f"Using manually set cycle bounds: {indices}")
    else:
        indices = detect_cycle_bounds(X_pca)
    np.savetxt(output_path + "_phase_indices.txt", indices, fmt="%i", delimiter=";")

    avg_signal_d = X_pca[indices[0] : indices[1]]
    M = len(avg_signal_d)
    avg_signal_d_av = np.zeros([M // REFERENCE_NUM_POINTS, 3])
    for i in range(0, M // REFERENCE_NUM_POINTS):
        avg_signal_d_av[i] = np.mean(avg_signal_d[i * REFERENCE_NUM_POINTS : (i + 1) * REFERENCE_NUM_POINTS], axis=0)
    smooth_points = smooth_trajectory(avg_signal_d_av)

    i = 0
    phase0 = np.array([], dtype=np.int32)
    prev_phase = 0

    if DO_UPDATE_REFERENCE_CYCLE:
        while i < X_pca.shape[0]:
            phaseh = assign_phase_indices(X_pca[indices[0] + i : indices[0] + i + UPDATE_REFERENCE_CYCLE_SIZE], smooth_points, prev_phase=prev_phase)
            smooth_points = update_reference_cycle(phaseh, smooth_points, X_pca[indices[0] + i : indices[0] + i + UPDATE_REFERENCE_CYCLE_SIZE])
            i += UPDATE_REFERENCE_CYCLE_SIZE
            phase0 = np.concatenate((phase0, phaseh))
            prev_phase = phaseh[-1]
    else:
        phase0 = assign_phase_indices(X_pca[indices[0] :], smooth_points)

    phase = phase0 / len(smooth_points) * 2 * np.pi
    phase = np.unwrap(phase)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(321, projection="3d")
    ax.plot(*smooth_points.T, "-", linewidth=2, label="Smoothed Loop", color="blue")
    ax.plot(*avg_signal_d.T, "-", linewidth=2, label="Points", color="red")

    ax = fig.add_subplot(322)
    ax.plot(X_pca[indices[0] : indices[0] + 20000, 0], color="black")
    ax.plot(smooth_points[phase0[:20000], 0])

    ax = fig.add_subplot(323)
    ax.plot(signals[indices[0] : indices[0] + 20000])

    ax = fig.add_subplot(324)
    ax.plot(phase, label="Phase")

    ax = fig.add_subplot(313)
    ax.plot(X_pca[indices[0] : 2 * indices[1] - indices[0], 0], color="black")
    ax.plot(avg_signal_d[:, 0], color="red")
    xticks = np.linspace(0, 2 * indices[1] - 2 * indices[0], 40)
    xticks = xticks.astype(int)
    xticks = 100 * np.round(xticks / 100).astype(int)
    ax.set_xticks(xticks)
    labels = [str(indices[0] + xt) for xt in xticks]
    ax.set_xticklabels(labels, rotation=90)
    for tick in ax.get_xticks():
        ax.axvline(x=tick, color="gray", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path + "_phase_debug.png")

    np.save(output_path + "_phase.npy", phase)
    ## np.save(output_path + "_smooth_points.npy", smooth_points[phase0, 0])
    ## np.save(output_path + "_avg_signal.npy", X_pca[indices[0] :, 0])
