"""
Visualize X + iY anisotropy from the first NUM_SAMPLES of a TDMS file.
Useful for checking signal quality and harmonic structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile

# === Settings ===
tdms_path = "data/2025.04.15 patricia/file7.tdms"
NUM_SAMPLES = 100_000  # number of points to use

# === Load and extract first NUM_SAMPLES samples ===
tdms_file = TdmsFile.read(tdms_path)
group = tdms_file.groups()[0]
channels = group.channels()

C90 = np.array(channels[0].data)[:NUM_SAMPLES]
C45 = np.array(channels[1].data)[:NUM_SAMPLES]
C135 = np.array(channels[2].data)[:NUM_SAMPLES]
C0 = np.array(channels[3].data)[:NUM_SAMPLES]

# === Compute anisotropy ===
X = (C0 - C90) / (C0 + C90)
Y = (C45 - C135) / (C45 + C135)

# === Plot X + iY ===
if __name__ == "__main__":
    Z = X + 1j * Y

    plt.figure(figsize=(6, 6))
    plt.plot(Z.real, Z.imag, '.', alpha=0.2, markersize=1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Anisotropy: X + iY")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
