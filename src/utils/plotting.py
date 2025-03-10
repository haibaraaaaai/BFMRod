import matplotlib.pyplot as plt
import numpy as np


def plot_3d_trajectory(X_pca, smooth_points, output_path=None):
    """
    Plots the PCA-transformed 3D trajectory and the smoothed reference cycle.

    Args:
        X_pca (np.ndarray): PCA-transformed trajectory data.
        smooth_points (np.ndarray): Smoothed reference cycle.
        output_path (str, optional): If provided, saves the plot to this path.
    """
    fig = plt.figure(figsize=(6, 5), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*X_pca.T, "-", linewidth=1, label="PCA Trajectory", color="black")
    ax.plot(*smooth_points.T, "-", linewidth=2, label="Smoothed Loop", color="blue")
    ax.set_title("3D PCA Trajectory")
    ax.legend()
    if output_path:
        plt.savefig(output_path)
    plt.show()


def plot_pca_component(X_pca, phase_indices, smooth_points, output_path=None):
    """
    Plots the first PCA component against its smoothed reference trajectory.

    Args:
        X_pca (np.ndarray): PCA-transformed trajectory data.
        phase_indices (np.ndarray): Phase indices assigned to trajectory points.
        smooth_points (np.ndarray): Smoothed reference cycle.
        output_path (str, optional): If provided, saves the plot to this path.
    """
    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(X_pca[:, 0], color="black", label="PCA Component 0")

    # Ensure phase indices are mapped within the range of smooth_points
    normalized_phase_indices = np.mod(phase_indices, smooth_points.shape[0])
    plt.plot(smooth_points[normalized_phase_indices, 0], color="blue", linestyle="dashed", label="Smoothed Reference")

    plt.xlabel("Time Index")
    plt.ylabel("PCA Component 0")
    plt.legend()
    plt.title("PCA Component 0 vs. Smoothed Reference")

    if output_path:
        plt.savefig(output_path)
    plt.show()


def plot_phase_evolution(phase, output_path=None):
    """
    Plots the phase evolution over time.

    Args:
        phase (np.ndarray): Phase values.
        output_path (str, optional): If provided, saves the plot to this path.
    """
    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(phase, label="Phase", color="green")
    plt.xlabel("Time Index")
    plt.ylabel("Phase (radians)")
    plt.title("Phase Evolution Over Time")
    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(output_path)
    plt.show()


def plot_frequency(time, frequency, output_path=None):
    """
    Plots the instantaneous frequency over time.

    Args:
        time (np.ndarray): Time values.
        frequency (np.ndarray): Frequency values (Hz).
        output_path (str, optional): If provided, saves the plot to this path.
    """
    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(time, frequency, color="blue", label="Instantaneous Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Instantaneous Frequency vs. Time")
    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(output_path)
    plt.show()
