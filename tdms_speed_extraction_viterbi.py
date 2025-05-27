# === TDMS SPEED EXTRACTION WITH CUSTOM VITERBI DECODER ===

from nptdms import TdmsFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks, decimate, savgol_filter
from sklearn.mixture import GaussianMixture
import os
import matplotlib as mpl
from datetime import datetime
mpl.rcParams['agg.path.chunksize'] = 10000

# === CONFIGURATION ===
data_path = "data/2025.05.23 patricia ox"
tdms_filename = "file.tdms"
decimation_factor = 100
fs_original = 250000
fs = fs_original // decimation_factor
window_size = 100
overlap = 50
nperseg = 100
nfft = 400
save_plots = True
save_segment_plots = True#False
plot_stride = 100
show_plots = False
save_minute_segments = True

# === TDMS DATA LOADING ===
tdms_file = TdmsFile.read(os.path.join(data_path, tdms_filename))
group_name = tdms_file.groups()[0].name
group = tdms_file[group_name]

channel_map = {
    "PXI1Slot2/ai0": "C90",
    "PXI1Slot2/ai1": "C45",
    "PXI1Slot2/ai2": "C135",
    "PXI1Slot2/ai3": "C0"
}

data_dict = {}
for original, alias in channel_map.items():
    signal = group[original].data
    signal_dec = decimate(signal, decimation_factor, ftype='fir')
    data_dict[alias] = signal_dec

timestamps = group[list(channel_map.keys())[0]].time_track()
timestamps_dec = decimate(timestamps, decimation_factor, ftype='fir')
data_dict["Timestamp (s)"] = timestamps_dec

df = pd.DataFrame(data_dict)
segment_indices = list(range(0, (len(df) - window_size) // overlap))
print(f"➔ Analyzing {len(segment_indices)} segments across full trace.")

df["Itot"] = df[["C0", "C45", "C90", "C135"]].sum(axis=1)
df["X"] = (df["C0"] - df["C90"]) / (df["C0"] + df["C90"])
df["Y"] = (df["C45"] - df["C135"]) / (df["C45"] + df["C135"])
df["X+iY"] = df["X"] + 1j * df["Y"]

def compute_fft(data, fs, nperseg, nfft, onesided=True):
    freqs, power = welch(data, fs=fs, nperseg=nperseg, nfft=nfft, return_onesided=onesided)
    power /= np.max(power)
    return freqs, power

dominant_frequencies = []
segment_times = []
output_dir = os.path.join(data_path, "results", "plots", "20250523_6_viterbi")
os.makedirs(output_dir, exist_ok=True)

# === Segment-wise dominant frequency analysis with harmonic correction ===
for idx in segment_indices:
    start = idx * overlap
    end = start + window_size
    if end > len(df):
        continue  # Skip incomplete segments at the end of the trace

    # Extract segment-wise Itot signal and corresponding time
    segment_itot = df["Itot"].values[start:end]
    segment_time = df["Timestamp (s)"].values[start:end]
    seg_mid_time = np.mean(segment_time)  # Middle time of the segment for time alignment

    # Compute power spectral density (PSD) of Itot using Welch method
    freqs_itot, power_itot = compute_fft(segment_itot, fs, nperseg, nfft)

    # Identify the dominant peak in the power spectrum
    dominant_freq = freqs_itot[np.argmax(power_itot)]

    # Define a frequency band centered at half of the dominant frequency
    half_range = (0.45 * dominant_freq, 0.55 * dominant_freq)

    # Search for secondary peaks in the power spectrum
    secondary_peaks, _ = find_peaks(power_itot)
    freq_secondary = freqs_itot[secondary_peaks]
    power_secondary = power_itot[secondary_peaks]

    # Check if a secondary peak suggests the dominant peak is a 2x harmonic
    likely_1x = False
    for f, p in zip(freq_secondary, power_secondary):
        if half_range[0] <= f <= half_range[1] and p > 0.5:
            likely_1x = True
            break

    # Apply correction if a 2x harmonic is suspected
    if likely_1x:
        corrected_freq = dominant_freq / 2
        print(f"Segment {idx}: Dominant freq = {dominant_freq:.2f} Hz → Adjusted to fundamental ≈ {corrected_freq:.2f} Hz (Likely 2x harmonic)")
    else:
        corrected_freq = dominant_freq
        print(f"Segment {idx}: Dominant freq = {dominant_freq:.2f} Hz → Accepted as fundamental")

    dominant_frequencies.append(corrected_freq)
    segment_times.append(seg_mid_time)

# === Smoothing ===
window_l = 51#51
polyn_ord = 2
smoothed_speed = savgol_filter(dominant_frequencies, window_length=window_l, polyorder=polyn_ord)

# === GMM + Custom Viterbi Decoder ===
bic_scores = []
gmm_models = []
covariance_t = 'diag' #diag, spherical
for k in range(1, 5): #10
    gmm_k = GaussianMixture(n_components=k, covariance_type=covariance_t, random_state=0)
    gmm_k.fit(smoothed_speed.reshape(-1, 1))
    bic_scores.append(gmm_k.bic(smoothed_speed.reshape(-1, 1)))
    gmm_models.append(gmm_k)
optimal_k = np.argmin(bic_scores) + 1
gmm = gmm_models[optimal_k - 1]
labels = gmm.predict(smoothed_speed.reshape(-1, 1))
means = gmm.means_.flatten()

# === Custom Viterbi Decoder ===
def viterbi_penalized(obs, means, vars, penalty, init_probs=None, transmat=None):
    T, K = len(obs), len(means)
    if init_probs is None:
        init_probs = np.full(K, 1.0 / K)
    if transmat is None:
        transmat = np.full((K, K), 1.0 / K)
    ll = np.array([ -0.5 * np.log(2 * np.pi * v) - 0.5 * ((obs - m)**2 / v) for m, v in zip(means, vars)]).T
    V, B = np.zeros((T, K)), np.zeros((T, K), int)
    V[0] = np.log(init_probs) + ll[0]
    for t in range(1, T):
        for s in range(K):
            trans_costs = [V[t-1, ps] + np.log(transmat[ps, s]) + ll[t, s] - (penalty if ps != s else 0) for ps in range(K)]
            V[t, s] = max(trans_costs)
            B[t, s] = np.argmax(trans_costs)
    path = np.zeros(T, int)
    path[-1] = np.argmax(V[-1])
    for t in reversed(range(1, T)):
        path[t-1] = B[t, path[t]]
    return path

emission_vars = gmm.covariances_.flatten()
stay_prob = 0.95
switch_prob = (1.0 - stay_prob) / (optimal_k - 1)
transmat = np.full((optimal_k, optimal_k), switch_prob)
np.fill_diagonal(transmat, stay_prob)
init_probs = np.full(optimal_k, 1.0 / optimal_k)
lambda_penalty = 500.0 #5.0, 10, 25, 50
custom_labels = viterbi_penalized(smoothed_speed, means, emission_vars, lambda_penalty, init_probs, transmat)
custom_trace = [means[l] for l in custom_labels]

# Optional dwell correction
min_dwell = 50#25, 5
corrected_labels = custom_labels.copy()
i = 0
while i < len(corrected_labels):
    current = corrected_labels[i]
    j = i
    while j < len(corrected_labels) and corrected_labels[j] == current:
        j += 1
    if j - i < min_dwell:
        corrected_labels[i:j] = [corrected_labels[i-1] if i > 0 else corrected_labels[j]] * (j - i)
    i = j
corrected_trace = [means[l] for l in corrected_labels]

# === Plot and Save ===
plt.figure(figsize=(10, 4))
plt.plot(segment_times, dominant_frequencies, label="Original Speed", alpha=0.4)
plt.plot(segment_times, smoothed_speed, label="Smoothed", linewidth=2)
plt.plot(segment_times, corrected_trace, label="Corrected Viterbi", linestyle='--')
for m in sorted(means):
    plt.axhline(m, linestyle=':', color='gray')
    plt.text(segment_times[0], m, f"Level {m:.2f} Hz", fontsize=8)
plt.title("Speed Trace with Custom Viterbi")
plt.xlabel("Time (s)")
plt.ylabel("Speed (Hz)")
plt.grid(True)
plt.legend()
plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join(output_dir, "custom_viterbi_speed_trace.png"), dpi=300)
if show_plots:
    plt.show()

speed_df = pd.DataFrame({
    "Time (s)": segment_times,
    "Speed (Hz)": dominant_frequencies,
    "Smoothed Speed (Hz)": smoothed_speed,
    "Corrected Viterbi Level": corrected_trace
})
speed_df.to_csv(os.path.join(output_dir, "speed_trace_summary.csv"), index=False)
print("✅ Custom Viterbi summary saved.")

# === Save 1-minute window breakdowns for Custom Viterbi ===
if save_plots and save_minute_segments:
    total_duration = segment_times[-1] - segment_times[0]
    start_time = segment_times[0]
    end_time = segment_times[-1]
    minute_edges = np.arange(start_time, end_time, 60)
    for i in range(len(minute_edges)):
        seg_start = minute_edges[i]
        seg_end = minute_edges[i+1] if i + 1 < len(minute_edges) else end_time
        mask = (np.array(segment_times) >= seg_start) & (np.array(segment_times) < seg_end)
        if np.sum(mask) < 2:
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(np.array(segment_times)[mask], np.array(smoothed_speed)[mask], color='orange', label="Smoothed Speed")
        plt.plot(np.array(segment_times)[mask], np.array(corrected_trace)[mask], linestyle='--', color='black', label="Corrected Viterbi Level")
        for mean in sorted(means):
            plt.axhline(mean, linestyle=':', color='gray', alpha=0.6)
            plt.text(np.array(segment_times)[mask][0], mean, f"{mean:.2f} Hz", fontsize=7, color='gray', verticalalignment='bottom')
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (Hz)")
        plt.title(f"Speed Trace Subplot: Minute {i+1}")
        plt.legend()
        plt.tight_layout()
        minute_plot_path = os.path.join(output_dir, f"custom_viterbi_speed_trace_minute_{i+1}.png")
        plt.savefig(minute_plot_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {minute_plot_path}")

# === Save histogram of Smoothed Speed + Custom Viterbi Levels ===
plt.figure(figsize=(10, 5))
plt.hist(smoothed_speed, bins=40, alpha=0.6, color="lightgray", edgecolor="k")
for mean in sorted(set(corrected_trace)):
    plt.axvline(mean, color="black", linestyle="--", label=f"Viterbi Level @ {mean:.2f} Hz")
plt.title("Histogram of Smoothed Speed + Custom Viterbi Levels")
plt.xlabel("Speed (Hz)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "custom_viterbi_levels_histogram.png"), dpi=300)
print("✅ Saved Custom Viterbi histogram plot.")

# === Save optional per-segment diagnostic plots ===
if save_plots and save_segment_plots:
    for idx in segment_indices:
        if idx % plot_stride != 0:
            continue
        start = idx * overlap
        end = start + window_size
        if end > len(df):
            continue

        segment_itot = df["Itot"].values[start:end]
        segment_xiy = df["X+iY"].values[start:end]
        segment_x = df["X"].values[start:end]
        segment_y = df["Y"].values[start:end]
        segment_time = df["Timestamp (s)"].values[start:end]

        freqs_itot, power_itot = compute_fft(segment_itot, fs, nperseg, nfft)
        freqs_xiy, power_xiy = compute_fft(segment_xiy, fs, nperseg, nfft, onesided=False)

        fig = plt.figure(figsize=(14, 10))

        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax1.plot(freqs_itot, power_itot, label="Itot", color="green")
        ax1.plot(freqs_xiy, power_xiy, label="X+iY", color="blue", linestyle="dashed")
        peak_itot = freqs_itot[np.argmax(power_itot)]
        peak_xiy = freqs_xiy[np.argmax(power_xiy)]
        ax1.scatter(peak_itot, np.max(power_itot), color="red")
        ax1.annotate(f"{peak_itot:.1f} Hz", (peak_itot, np.max(power_itot)), textcoords="offset points", xytext=(5,5), color="red")
        ax1.scatter(peak_xiy, np.max(power_xiy), color="purple")
        ax1.annotate(f"{peak_xiy:.1f} Hz", (peak_xiy, np.max(power_xiy)), textcoords="offset points", xytext=(5,-10), color="purple")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Normalized Power")
        ax1.set_title(f"Segment {idx}: X+iY & Itot FFT")
        ax1.legend()

        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax2.plot(segment_x, segment_y, color='lightgray', linestyle='-', linewidth=1, alpha=0.5)
        ax2.scatter(segment_x, segment_y, color='teal', alpha=0.4, s=10)
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title(f"XY Scatter - Segment {idx}")
        ax2.grid(True, linestyle='--', linewidth=0.5)

        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        for ch in ["C0", "C45", "C90", "C135"]:
            ax3.plot(segment_time, df[ch].values[start:end], label=ch)
        ax3.plot(segment_time, segment_itot, label="Itot", linestyle="dashed", linewidth=1.5, color='purple')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Intensity")
        ax3.set_title(f"Raw Channel Data - Segment {idx}")
        ax3.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"segment_{idx}_multipanel.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {plot_path}")
