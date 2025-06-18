import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, decimate, savgol_filter
from nptdms import TdmsFile

# === Step Fitting Utilities ===

def get_variance(data, inter, pos):
    attempt = inter[:np.searchsorted(inter, pos)] + [pos] + inter[np.searchsorted(inter, pos):]
    delta = list(data[:attempt[0]] - np.mean(data[:attempt[0]]))
    for i in range(len(attempt) - 1):
        seg = data[attempt[i]:attempt[i + 1]]
        if len(seg) > 0:
            delta += list(seg - np.mean(seg))
    delta += list(data[attempt[-1]:] - np.mean(data[attempt[-1]:]))
    return np.mean(np.array(delta) ** 2.)

def get_pos(data, inter, res=50):
    variance = np.inf
    best_pos = None
    for pos in range(res, len(data) - res, res):
        if pos not in inter and 0 < pos < len(data):
            v = get_variance(data, inter, pos)
            if v < variance:
                variance = v
                best_pos = pos
    if best_pos is not None:
        inter = inter[:np.searchsorted(inter, best_pos)] + [best_pos] + inter[np.searchsorted(inter, best_pos):]
    return best_pos, inter, variance

def get_int_av(data, res=5, th=10, limit_ratio=0.99):
    inter = [0]
    variance = 100
    v_random = 200
    real = 10
    max_len = len(data)
    step_count = 0

    print("⏳ Starting step fitting...")

    while variance / v_random < limit_ratio:
        v_random = np.mean([
            get_variance(data, inter, np.random.randint(0, max_len))
            for _ in range(real)
        ])
        best_pos, inter, variance = get_pos(data, inter, res=res)
        step_count += 1

        if best_pos is None:
            print(f"✅ No more valid step positions. Total steps: {step_count}")
            break

        print(f"  ➤ Step {step_count:2d}: added pos={best_pos}, variance={variance:.4f}, ratio={variance / v_random:.4f}")

    inter = sorted(set(inter))
    if inter[-1] != max_len - 1:
        inter.append(max_len - 1)

    step_fitted = np.zeros_like(data)
    for i in range(len(inter) - 1):
        start, end = inter[i], inter[i + 1]
        if end > start:
            avg = np.mean(data[start:end])
            step_fitted[start:end] = avg
        else:
            print(f"⚠️ Skipping invalid segment: start={start}, end={end}")

    print("✅ Step fitting complete.")
    return step_fitted

# === Parameters ===
data_path = "data/2024.10.17 daping ox"
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

# === Process all .tdms files in data_path ===
for tdms_filename in sorted(f for f in os.listdir(data_path) if f.endswith(".tdms")):
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

    smoothed_speeds = savgol_filter(speeds, window_length=51, polyorder=2)
    step_fitted = get_int_av(smoothed_speeds, res=5, th=10)

    plt.figure(figsize=(10, 5))
    plt.plot(segment_times, speeds, label="Raw Speed", alpha=0.3, marker='o', markersize=3)
    plt.plot(segment_times, smoothed_speeds, label="Savitzky-Golay", linewidth=2, color='orange')
    plt.plot(segment_times, step_fitted, label="Step-Fitted", linewidth=2, linestyle='--', color='green')
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (Hz)")
    plt.title("Step-Fitted Speed Trace from Itot")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "itot_speed_trace_stepfit.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ Step-fitted speed plot saved: {plot_path}")

    df_speed = pd.DataFrame({
        "Time (s)": segment_times,
        "Speed (Hz)": speeds,
        "Smoothed Speed (Hz)": smoothed_speeds,
        "Step-Fitted Speed (Hz)": step_fitted
    })
    df_speed.to_csv(os.path.join(output_dir, "itot_speed_trace_stepfit.csv"), index=False)
    print("✅ Speed data saved to CSV.\n")
