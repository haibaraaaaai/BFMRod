import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile

# === Settings ===
tdms_path = "data/2025.03.20 patricia/file8.tdms"
N = 100_000  # number of points to use

# === Load and extract first N samples ===
tdms_file = TdmsFile.read(tdms_path)
group = tdms_file.groups()[0]
channels = group.channels()

C90 = np.array(channels[0].data)[:N]
C45 = np.array(channels[1].data)[:N]
C135 = np.array(channels[2].data)[:N]
C0 = np.array(channels[3].data)[:N]

# === Compute anisotropy ===
with np.errstate(divide='ignore', invalid='ignore'):
    X = (C0 - C90) / (C0 + C90)
    Y = (C45 - C135) / (C45 + C135)

# Optional: remove invalid values (from divide by zero)
mask = np.isfinite(X) & np.isfinite(Y)
X = X[mask]
Y = Y[mask]

# === Plot X + iY ===
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
