import numpy as np
import pandas as pd
from scipy.signal import welch
from nptdms import TdmsFile

# Parameters
data_path = "data/20250609/files8.tdms"
fs = 250000
window_size = 8000
overlap = 4000
nperseg = 8000
nfft = 32000

channel_map = {
    "PXI1Slot2/ai0": "C90",
    "PXI1Slot2/ai1": "C45",
    "PXI1Slot2/ai2": "C135",
    "PXI1Slot2/ai3": "C0"
}

# Load TDMS data
tdms_file = TdmsFile.read(data_path)
group_name = tdms_file.groups()[0].name
group = tdms_file[group_name]

# Extract and sum channel signals into Itot
itot = None
for tdms_name in channel_map.keys():
    signal = group[tdms_name].data
    itot = signal if itot is None else itot + signal

# Extract time base
timestamps = group[list(channel_map.keys())[0]].time_track()

# Compute FFT-based speed from overlapping windows
step = window_size - overlap
segment_times = []
dominant_freqs = []

for start in range(0, len(itot) - window_size + 1, step):
    end = start + window_size
    segment = itot[start:end]
    time_segment = timestamps[start:end]
    mid_time = np.mean(time_segment)

    freqs, power = welch(segment, fs=fs, nperseg=nperseg, nfft=nfft)
    dominant_freq = freqs[np.argmax(power)]

    segment_times.append(mid_time)
    dominant_freqs.append(dominant_freq)

# Save result to DataFrame
df = pd.DataFrame({
    "Time (s)": segment_times,
    "Speed (Hz)": dominant_freqs
})

# Apply FIR smoothing with 100-point moving average
window_size_smooth = 100
kernel = np.ones(window_size_smooth) / window_size_smooth
smoothed_speed = np.convolve(df["Speed (Hz)"], kernel, mode="valid")
df = df.iloc[window_size_smooth - 1:].copy()
df["Smoothed Speed (Hz)"] = smoothed_speed

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import ruptures as rpt

# Step 1: Apply step fitting using KernelCPD (get_ruptures_mm logic)
penalty = 70
model = rpt.KernelCPD(kernel="linear").fit(df["Smoothed Speed (Hz)"].values)
xbound = model.predict(pen=penalty)
xbound = [0] + xbound

m = []
for i in range(len(xbound) - 1):
    start, end = xbound[i], xbound[i + 1]
    segment = df["Smoothed Speed (Hz)"].values[start:end]
    m.append(np.mean(segment))
m = np.array(m)

# Step 2: Define KDE states
kde = gaussian_kde(df["Smoothed Speed (Hz)"].values, bw_method=0.03)
s_vals = np.linspace(min(m), max(m), 2000)
pdf = kde(s_vals)
prominence = np.max(pdf) * 0.01
peaks, _ = find_peaks(pdf, prominence=prominence)
states = s_vals[peaks]
indli = np.array([np.argmin(np.abs(states - val)) for val in m])

# Step 3: Compute transition statistics
transitionarh = {}
timessh = {}
rawtrh = {}
meanzh = {}
durationsh = {}

for i in range(len(indli)-1):
    fr = indli[i]
    to = indli[i+1]
    key = (fr, to)
    t_start = df["Time (s)"].iloc[xbound[i]]
    t_end = df["Time (s)"].iloc[xbound[i+1]]
    speed_segment = df["Speed (Hz)"].iloc[xbound[i]:xbound[i+1]]

    if key not in transitionarh:
        transitionarh[key] = []
        timessh[key] = []
        rawtrh[key] = []
        meanzh[key] = []
        durationsh[key] = []

    transitionarh[key].append(t_end)
    timessh[key].append(t_end - t_start)
    rawtrh[key].append(speed_segment.values)
    meanzh[key].append(np.mean(speed_segment))
    durationsh[key].append(t_end - t_start)

# Cleanup loop with bounds check
for i in range(1, len(xbound)):
    if xbound[i] >= len(df) or xbound[i-1] >= len(df):
        continue
    t_prev = df["Time (s)"].iloc[xbound[i-1]]
    t_now = df["Time (s)"].iloc[xbound[i]]

# Print summary
print("Transitions between KDE states:")
for key in sorted(transitionarh):
    print(f"{key}: count={len(transitionarh[key])}, avg dwell={np.mean(durationsh[key]):.2f}s")

import matplotlib.pyplot as plt

# Step 4: Clean up short dwells and reversions
min_dwell_time = 0.2  # seconds
xbound_clean = [xbound[0]]
indli_clean = [indli[0]]

for i in range(1, len(xbound)):
    if xbound[i] >= len(df) or xbound[i-1] >= len(df):
        continue  # skip out-of-bounds indices
    t_prev = df["Time (s)"].iloc[xbound[i-1]]
    t_now = df["Time (s)"].iloc[xbound[i]]
    dwell_time = t_now - t_prev
    current_state = indli[i]

    if dwell_time < min_dwell_time and current_state == indli_clean[-1]:
        continue  # merge into previous segment (self-reversion)
    xbound_clean.append(xbound[i])
    indli_clean.append(current_state)

# Step 5: Plot original and cleaned state trace
state_trace = np.repeat(indli, np.diff(xbound))
state_trace_clean = np.repeat(indli_clean[:-1], np.diff(xbound_clean))

plt.figure(figsize=(12, 4))
plt.plot(df["Time (s)"], df["Smoothed Speed (Hz)"], label="Smoothed Speed", alpha=0.5)
plt.step(df["Time (s)"][:len(state_trace)], states[state_trace], label="Original States", linewidth=1.5)
plt.step(df["Time (s)"][:len(state_trace_clean)], states[state_trace_clean], label="Cleaned States", linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Speed (Hz)")
plt.legend()
plt.title("State Assignment Before and After Cleanup")
plt.tight_layout()
plt.show()
