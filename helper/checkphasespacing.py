import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from nptdms import TdmsFile

# ========== CONFIG ========== #
REFERENCE_NUM_POINTS = 200
SMOOTHING = 0.001  # B-spline smoothing factor
TDMS_FILE = "data/2025.03.20 patricia/file8.tdms"  # 🔁 Replace with your file path
REF_START = 1730  # 🔁 Replace with your reference start index
REF_END = 4141    # 🔁 Replace with your reference end index

# ========== FUNCTIONS ========== #
def smooth_trajectory(points, smoothing_factor=SMOOTHING, num_points=REFERENCE_NUM_POINTS):
    """
    Apply cubic B-spline interpolation to smooth a closed 3D trajectory.
    """
    points = np.array(points)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    tck, _ = splprep(points.T, per=True, s=smoothing_factor)
    u_fine = np.linspace(0, 1, num_points)
    smooth_points = np.array(splev(u_fine, tck)).T
    return smooth_points

def compute_pca(data):
    data_centered = data - np.mean(data, axis=0)
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    return data_centered @ Vt.T

# ========== LOAD DATA ========== #
tdms_file = TdmsFile.read(TDMS_FILE)
group = tdms_file.groups()[0]
channels = group.channels()

# === Load raw channel data ===
C90 = np.array(channels[0].data)[:250_000]
C45 = np.array(channels[1].data)[:250_000]
C135 = np.array(channels[2].data)[:250_000]
C0 = np.array(channels[3].data)[:250_000]

# === Compute anisotropy ===
with np.errstate(divide='ignore', invalid='ignore'):
    X = (C0 - C90) / (C0 + C90)
    Y = (C45 - C135) / (C45 + C135)

mask = np.isfinite(X) & np.isfinite(Y)
X = X[mask]
Y = Y[mask]
Z = X + 1j * Y

# === PCA ===
raw_data = np.stack([C0, C45, C90, C135], axis=1)
pca_data = compute_pca(raw_data)[:, :3]

# === Reference cycle extraction and smoothing ===
initial_cycle = pca_data[REF_START:REF_END]
smooth_ref_cycle = smooth_trajectory(initial_cycle)

# === Find closest PCA indices for smoothed ref points ===
tree = cKDTree(pca_data)
_, matched_indices = tree.query(smooth_ref_cycle, k=1)

# === Extract Z at matched reference points ===
Z_ref = Z[matched_indices]

# ========== PLOT Z-SPACE ONLY (with highlights) ========== #
plt.figure(figsize=(6, 6))
plt.plot(Z.real, Z.imag, '.', alpha=0.1, label="Z raw data")

# Subsample every 5th ref point
Z_ref_sub = Z_ref[::5]

# Plot all ref points
plt.plot(Z_ref_sub.real, Z_ref_sub.imag, 'ro', label="Z ref points (every 5)")
for i, z in enumerate(Z_ref_sub):
    idx = i * 5
    plt.text(z.real, z.imag, str(idx), fontsize=8, ha='center', va='center')

# Highlight key points: 0, 10, 15
highlight_indices = [0, 10, 15]
highlight_colors = ['blue', 'green', 'magenta']
for hi, color in zip(highlight_indices, highlight_colors):
    z = Z_ref[hi]
    plt.plot(z.real, z.imag, 'o', markersize=10, color=color, label=f"Point {hi}")
    plt.text(z.real, z.imag, str(hi), fontsize=9, fontweight='bold',
             ha='center', va='center', color='white')

plt.xlabel("X = (C0 - C90) / (C0 + C90)")
plt.ylabel("Y = (C45 - C135) / (C45 + C135)")
plt.title("Z = X + iY (Anisotropy Space)")
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()