import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate, savgol_filter

# Define constants
PHASE_FILE = "results/2025.02.23 ox/file1_0_60_phase.npy"  # Path to saved phase data
SAMPLING_RATE = 1 / 250000  # 250 kHz = 1/250000 seconds per sample

# Data processing parameters
DECIMATION_FACTOR = 100  # Downsampling factor (reduces data points by 100x)
SMOOTHING_WINDOW = 51  # Window size for Savitzky-Golay filter (must be odd)

def compute_frequency(phase, sampling_rate):
    """
    Compute instantaneous frequency (Hz) from phase data.

    The frequency is derived by differentiating the phase and 
    converting from angular velocity (radians/s) to Hz.

    Args:
        phase (np.ndarray): Phase values in radians.
        sampling_rate (float): Time interval between phase points (seconds).

    Returns:
        np.ndarray: Computed frequency values in Hz.
        np.ndarray: Corresponding time values.
    """
    time = np.arange(len(phase)) * sampling_rate  # Generate time array
    speed = np.gradient(phase) / sampling_rate  # Compute angular velocity (radians/s)
    frequency = speed / (2 * np.pi)  # Convert to Hz (f = ω / 2π)
    return time, frequency

def process_frequency(frequency, time, decimation_factor, smoothing_window):
    """
    Decimate and smooth frequency data for better visualization.

    This function reduces the number of data points via decimation 
    and applies a Savitzky-Golay filter for smoothing.

    Args:
        frequency (np.ndarray): Instantaneous frequency values (Hz).
        time (np.ndarray): Corresponding time values.
        decimation_factor (int): Factor by which to reduce data points.
        smoothing_window (int): Window size for Savitzky-Golay filter.

    Returns:
        np.ndarray: Processed frequency values.
        np.ndarray: Processed time values.
    """
    # Decimate data to reduce size (applies anti-aliasing filter)
    frequency_decimated = decimate(frequency, decimation_factor, zero_phase=True)
    time_decimated = time[::decimation_factor]  # Downsample time array

    # Apply Savitzky-Golay filter to smooth high-frequency noise
    frequency_smoothed = savgol_filter(frequency_decimated, smoothing_window, polyorder=3)

    return time_decimated, frequency_smoothed

def plot_frequency(time, frequency, output_path):
    """
    Plot instantaneous frequency over time.

    Args:
        time (np.ndarray): Time values.
        frequency (np.ndarray): Frequency values (Hz).
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(time, frequency, color="blue", label="Instantaneous Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Instantaneous Frequency vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path + "_frequency_plot.png")
    plt.show()

if __name__ == "__main__":
    # Check if phase data file exists
    if not os.path.exists(PHASE_FILE):
        print(f"Error: File {PHASE_FILE} not found.")
        exit(1)

    # Generate output file path (remove .npy extension)
    output_path = os.path.splitext(PHASE_FILE)[0]

    # Load phase data from file
    phase = np.load(PHASE_FILE)

    # Compute instantaneous frequency in Hz
    time, frequency = compute_frequency(phase, SAMPLING_RATE)

    # Process frequency (decimate + smooth)
    time_processed, frequency_processed = process_frequency(frequency, time, DECIMATION_FACTOR, SMOOTHING_WINDOW)

    # Plot processed frequency
    plot_frequency(time_processed, frequency_processed, output_path)
