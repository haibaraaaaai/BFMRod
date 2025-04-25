import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate, savgol_filter

# Define constants
PHASE_FILE = "results/2025.02.23 ox/file1_0_60_phase.npy"  # Phase data path
SAMPLING_RATE = 1 / 250000  # 250 kHz sampling interval

# Data processing parameters
DECIMATION_FACTOR = 100  # Downsampling factor
SMOOTHING_WINDOW = 51  # Savitzky-Golay filter window size (odd)

def compute_frequency(phase, sampling_rate):
    """Convert phase to frequency (Hz) over time."""
    time = np.arange(len(phase)) * sampling_rate
    speed = np.gradient(phase) / sampling_rate
    frequency = speed / (2 * np.pi)
    return time, frequency

def process_frequency(frequency, time, decimation_factor, smoothing_window):
    """Downsample and smooth the frequency trace."""
    frequency_decimated = decimate(frequency, decimation_factor, zero_phase=True)
    time_decimated = time[::decimation_factor]
    frequency_smoothed = savgol_filter(frequency_decimated, smoothing_window, polyorder=3)
    return time_decimated, frequency_smoothed

def plot_frequency(time, frequency, output_path):
    """Plot frequency vs. time and save as PNG."""
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

    # Set output base path
    output_path = os.path.splitext(PHASE_FILE)[0]

    # Load phase
    phase = np.load(PHASE_FILE)

    # Compute frequency
    time, frequency = compute_frequency(phase, SAMPLING_RATE)

    # Decimate and smooth
    time_processed, frequency_processed = process_frequency(frequency, time, DECIMATION_FACTOR, SMOOTHING_WINDOW)

    # Plot result
    plot_frequency(time_processed, frequency_processed, output_path)
