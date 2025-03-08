import os
import numpy as np
import matplotlib.pyplot as plt

# Define constants
PHASE_FILE = "data/2025.02.23 ox/file1_phase.npy"  # Change this to the actual file path
SAMPLING_RATE = 1 / 250000  # 250 kHz = 1/250000 seconds per sample

def compute_speed(phase, sampling_rate):
    """
    Compute speed from phase data.
    
    Args:
        phase (np.ndarray): Phase values.
        sampling_rate (float): Time interval between phase points (in seconds).
    
    Returns:
        np.ndarray: Speed values (radians per second).
        np.ndarray: Corresponding time values.
    """
    print(len(phase))
    time = np.arange(len(phase)) * sampling_rate  # Generate time array
    speed = np.gradient(phase) / sampling_rate  # Compute speed (radians per second)
    return time, speed

def plot_speed(time, speed, output_path):
    """
    Plot speed over time.
    
    Args:
        time (np.ndarray): Time values.
        speed (np.ndarray): Speed values.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(time, speed, color="red", label="Speed")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (radians/s)")
    plt.title("Speed vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path + "_speed_plot.png")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(PHASE_FILE):
        print(f"Error: File {PHASE_FILE} not found.")
        exit(1)

    output_path = os.path.splitext(PHASE_FILE)[0]  # Remove .npy extension

    phase = np.load(PHASE_FILE)
    time, speed = compute_speed(phase, SAMPLING_RATE)
    plot_speed(time, speed, output_path)
