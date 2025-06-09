import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import ruptures as rpt
from scipy.signal import savgol_filter, decimate, find_peaks, welch
from scipy.stats import gaussian_kde
from nptdms import TdmsFile
import os
import pandas as pd

# Parameters
data_path = "data/2025.05.23 patricia ox"
decimation_factor = 100
fs_original = 250000
fs = fs_original // decimation_factor
window_size = 100
overlap = 50
nperseg = 100
nfft = 400

channel_map = {
    "PXI1Slot2/ai0": "C90",
    "PXI1Slot2/ai1": "C45",
    "PXI1Slot2/ai2": "C135",
    "PXI1Slot2/ai3": "C0"
}

tdms_filename = "files3.tdms"  # Replace with actual file name
print(f"📂 Processing: {tdms_filename}")

output_dir = os.path.join(data_path, os.path.splitext(tdms_filename)[0], "stepfit_speed_trace")
os.makedirs(output_dir, exist_ok=True)

tdms_file = TdmsFile.read(os.path.join(data_path, tdms_filename))
group_name = tdms_file.groups()[0].name
group = tdms_file[group_name]

data = {}
for tdms_name, alias in channel_map.items():
    signal = group[tdms_name].data
    signal_dec = decimate(signal, decimation_factor, ftype='fir')
    data[alias] = signal_dec

timestamps = group[list(channel_map.keys())[0]].time_track()
timestamps_dec = decimate(timestamps, decimation_factor, ftype='fir')
data["Timestamp (s)"] = timestamps_dec

df = pd.DataFrame(data)
df["Itot"] = df[["C0", "C45", "C90", "C135"]].sum(axis=1)

segment_indices = list(range(0, (len(df) - window_size) // overlap))
segment_times = []
speeds = []

for idx in segment_indices:
    start = idx * overlap
    end = start + window_size
    if end > len(df):
        continue
    segment = df["Itot"].values[start:end]
    time_segment = df["Timestamp (s)"].values[start:end]
    mid_time = np.mean(time_segment)
    freqs, power = welch(segment, fs=fs, nperseg=nperseg, nfft=nfft)
    dominant_freq = freqs[np.argmax(power)]
    segment_times.append(mid_time)
    speeds.append(dominant_freq)

t = np.array(segment_times)
s = np.array(speeds)

# --- Optional: smoothing the speed trace ---
# Savitzky-Golay filter works well for preserving step structure
window_length = 101  # Must be odd and less than the length of s
s_smooth = savgol_filter(s, window_length, polyorder=2)

# --- Preview KDE and speed plots for initial windows ---
window_duration = 20.0  # seconds
i_start = 0
n_preview = 4
preview_count = 0
plt.figure(figsize=(12, 2.5 * n_preview))
while i_start < len(t) and preview_count < n_preview:
    i_end = i_start
    while i_end < len(t) and t[i_end] - t[i_start] < window_duration:
        i_end += 1
    if i_end - i_start < 10:
        break

    local_t = t[i_start:i_end]
    local_raw = s[i_start:i_end]
    local_smooth = s_smooth[i_start:i_end]

    local_kde = gaussian_kde(local_smooth, bw_method=0.04)
    s_vals_local = np.linspace(local_smooth.min(), local_smooth.max(), 500)
    pdf_local = local_kde(s_vals_local)
    prom_local = np.max(pdf_local) * 0.2
    peaks_local, _ = find_peaks(pdf_local, prominence=prom_local)
    local_states = s_vals_local[peaks_local]

    ax = plt.subplot(n_preview, 1, preview_count + 1)
    ax2 = ax.twinx()
    ax.plot(s_vals_local, pdf_local, label="Local KDE", color="tab:blue")
    ax.plot(local_states, pdf_local[peaks_local], 'rx', label="Peaks")

    ax2.plot(local_t, local_raw, alpha=0.3, label="Raw", color="gray")
    ax2.plot(local_t, local_smooth, label="Smoothed", color="orange")

    ax.set_title(f"KDE + Speed for window starting at {t[i_start]:.2f}s")
    ax.set_ylabel("Density")
    ax2.set_ylabel("Speed (Hz)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    i_start = i_end
    preview_count += 1

plt.tight_layout()
plt.show()

# --- Segment-wise step fitting using 20s KDE windows ---
window_duration = 20.0  # seconds
segmentwise_step = np.copy(s_smooth)
i_start = 0

while i_start < len(t):
    i_end = i_start
    while i_end < len(t) and t[i_end] - t[i_start] < window_duration:
        i_end += 1
    if i_end - i_start < 10:
        break

    local_t = t[i_start:i_end]
    local_s = s_smooth[i_start:i_end]

    # Local KDE and peaks (stricter detection)
    local_kde = gaussian_kde(local_s, bw_method=0.04)
    s_vals_local = np.linspace(local_s.min(), local_s.max(), 500)
    pdf_local = local_kde(s_vals_local)
    prom_local = np.max(pdf_local) * 0.2
    peaks_local, _ = find_peaks(pdf_local, prominence=prom_local)
    local_states = s_vals_local[peaks_local]
    local_states.sort()

    # Local CPD and step fitting
    if len(local_states) > 0 and len(local_s) > 10:
        model = rpt.KernelCPD(kernel="linear").fit(local_s)
        cp_local = model.predict(pen=70)
        cp_local = [0] + cp_local

        for j in range(len(cp_local)-1):
            start, end = cp_local[j], cp_local[j+1]
            seg = local_s[start:end]
            avg = np.mean(seg)
            state = local_states[np.argmin(np.abs(local_states - avg))]
            segmentwise_step[i_start+start:i_start+end] = state

    i_start = i_end

# --- Sliding window KDE peak tracking ---
window_pts = 200
step_pts = 100
window_peaks_times = []
window_peaks_values = []

for i in range(0, len(s_smooth) - window_pts, step_pts):
    local_speed = s_smooth[i:i+window_pts]
    local_time = t[i + window_pts//2]
    local_kde = gaussian_kde(local_speed, bw_method=0.012)
    s_vals_local = np.linspace(local_speed.min(), local_speed.max(), 500)
    pdf_local = local_kde(s_vals_local)
    prom_local = np.max(pdf_local) * 0.2
    peaks_local, _ = find_peaks(pdf_local, prominence=prom_local)
    for pk in s_vals_local[peaks_local]:
        window_peaks_times.append(local_time)
        window_peaks_values.append(pk)

# --- Step 1: Estimate global states via KDE ---
kde = gaussian_kde(s_smooth, bw_method=0.03)  # you can adjust bw_method
# KDE is evaluated over this grid of values.
# Increasing the number (e.g., 1000 → 2000) makes the KDE plot smoother but slower to compute.
# Shrinking the range (e.g., using percentiles) can focus the KDE on the most relevant speed range.
s_vals = np.linspace(s_smooth.min(), s_smooth.max(), 2000)
pdf = kde(s_vals)

# Find peaks in the KDE (i.e., likely stable speed levels)
prominence = np.max(pdf) * 0.01  # % of max KDE height
peaks, _ = find_peaks(pdf, prominence)

global_states = s_vals[peaks]

num_peaks = len(peaks)
max_peak_height = np.max(pdf[peaks]) if num_peaks > 0 else 0
min_peak_height = np.min(pdf[peaks]) if num_peaks > 0 else 0
median_spacing = np.median(np.diff(global_states)) if num_peaks > 1 else 0

if len(global_states) > 0:
    # --- Step 2: Apply KernelCPD to detect candidate step locations ---
    model = rpt.KernelCPD(kernel="linear").fit(s_smooth)
    pen=100
    change_points = model.predict(pen)

    # --- Step 3: Map each segment to closest global state ---
    s_step = np.copy(s_smooth)
    prev_idx = 0
    for cp in change_points:
        segment = s_smooth[prev_idx:cp]
        avg_val = np.mean(segment)
        closest_state = global_states[np.argmin(np.abs(global_states - avg_val))]
        s_step[prev_idx:cp] = closest_state
        prev_idx = cp

# Plot peak trajectories over time
plt.figure(figsize=(12, 9))

# Top: Speed trace
plt.subplot(3, 1, 1)
plt.plot(t, s, alpha=0.3, label="Raw Speed")
plt.plot(t, s_smooth, label="Smoothed Speed", linewidth=1)
if len(global_states) > 0:
    plt.plot(t, s_step, label="Step Fitted", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Speed (Hz)")
plt.legend()
plt.title("Speed vs Time")

# Middle: Global KDE
plt.subplot(3, 1, 2)
plt.hist(s_smooth, bins=50, density=True, alpha=0.3, label="Histogram")
plt.plot(s_vals, pdf, label="KDE")
plt.plot(global_states, pdf[peaks], "rx", label="Peaks")
for peak_val in global_states:
    plt.axvline(peak_val, color="red", linestyle="--", alpha=0.5)
plt.xlabel("Speed (Hz)")
plt.ylabel("Density")
plt.legend()
plt.title(f"KDE and Detected Peaks (SavGol win={window_length}, bw={kde.factor:.3f}, prom={prominence:.2e}, peaks={num_peaks}, pen={pen})")

# Bottom: Sliding window peak map
plt.subplot(3, 1, 3)
plt.scatter(window_peaks_times, window_peaks_values, s=3, alpha=0.6)
plt.xlabel("Time (s)")
plt.ylabel("Local KDE Peak (Hz)")
plt.title("Sliding Window KDE Peak Trajectories")

plt.tight_layout()
plt.show()

# Comparison plot with global vs segment-wise step fitting
plt.figure(figsize=(12, 5))
plt.plot(t, s, alpha=0.2, label="Raw Speed")
plt.plot(t, s_smooth, label="Smoothed", linewidth=1)
if 's_step' in locals():
    plt.plot(t, s_step, label="Global Step Fit", linewidth=2, linestyle="--")
plt.plot(t, segmentwise_step, label="Windowed Step Fit", linewidth=2, linestyle=":")
plt.xlabel("Time (s)")
plt.ylabel("Speed (Hz)")
plt.title("Global vs Windowed Step-Fitted Traces")
plt.legend()
plt.tight_layout()
plt.show()