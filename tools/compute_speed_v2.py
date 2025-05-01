import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

# --- Settings ---
REFERENCE_NUM_POINTS = 200  # Used to convert phase0 to float phase

def compute_revolution_frequency(phase, phase_time, rev_step=2*np.pi):
    step = rev_step
    max_phase = phase[-1]
    thresholds = np.arange(step, max_phase, step)

    rev_times = []
    for threshold in thresholds:
        idx = np.searchsorted(phase, threshold)
        if idx >= len(phase_time):
            break
        rev_times.append(phase_time[idx])

    rev_times = np.array([phase_time[0]] + rev_times)

    freq_times = []
    freq_values = []
    for i in range(len(rev_times) - 1):
        t_start = rev_times[i]
        t_end = rev_times[i + 1]
        duration = t_end - t_start
        if duration < 0.001:  # skip too short revs (likely artifact from bad initial phase)
            continue
        freq = 1 / duration

        mid1 = i + (1 - 1) // 2
        mid2 = i + (1 + 1) // 2
        freq_time = (rev_times[mid1] + rev_times[mid2]) / 2

        freq_times.append(freq_time)
        freq_values.append(freq)

    return np.array(freq_times), np.array(freq_values)

def get_variance(data, inter, pos):
    attempt = inter[:np.searchsorted(inter, pos)] + [pos] + inter[np.searchsorted(inter, pos):]
    delta = list(data[:attempt[0]] - np.mean(data[:attempt[0]]))
    for i in range(len(attempt) - 1):
        delta += list(data[attempt[i]:attempt[i + 1]] - np.mean(data[attempt[i]:attempt[i + 1]]))
    delta += list(data[attempt[-1]:] - np.mean(data[attempt[-1]:]))
    return np.mean(np.array(delta) ** 2.)

def get_pos(data, inter, res=50):
    variance = np.inf
    best_pos = None
    for pos in range(res, len(data) - res, res):
        if pos not in inter:
            v = get_variance(data, inter, pos)
            if v < variance:
                variance = v
                best_pos = pos
    if best_pos is not None:
        inter = inter[:np.searchsorted(inter, best_pos)] + [best_pos] + inter[np.searchsorted(inter, best_pos):]
    return best_pos, inter, variance

def get_int_av(data, res, th=10, limit_ratio=0.99):
    inter = [0]
    variance = 100
    v_random = 200
    real = 10
    while variance / v_random < limit_ratio:
        v_random = np.mean([get_variance(data, inter, np.random.randint(0, len(data))) for _ in range(real)])
        best_pos, inter, variance = get_pos(data, inter, res=res)
        if best_pos is None:
            break
    inter.append(len(data) - 1)
    averages = [np.mean(data[inter[i]:inter[i + 1]]) for i in range(len(inter) - 1)]
    step_fitted = np.zeros_like(data)
    for i in range(len(inter) - 1):
        step_fitted[inter[i]:inter[i + 1]] = averages[i]
    return step_fitted

def analyze_file(npz_path):
    try:
        data = np.load(npz_path)
        phase0 = data["phase0"]
        phase_time = data["phase_time"]

        cache_path = os.path.splitext(npz_path)[0] + "_speeds.npz"
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            t_ref = cached["t_ref"]
            s_ref = cached["s_ref"]
            step_fitted = cached["step_fitted"]
            print(f"Loaded cached data from {cache_path}")
        else:
            phase_raw = np.unwrap(phase0 / REFERENCE_NUM_POINTS * 2 * np.pi)
            phase = savgol_filter(phase_raw, window_length=2001, polyorder=3)
            t_ref, s_ref = compute_revolution_frequency(phase, phase_time)
            step_fitted = get_int_av(s_ref, res=5, th=10)

            np.savez(cache_path, t_ref=t_ref, s_ref=s_ref, step_fitted=step_fitted)
            print(f"Saved computed speed and step data to: {cache_path}")

        # Plot
        plt.figure(figsize=(12, 4))
        plt.plot(t_ref, s_ref, label="Smoothed Speed", alpha=0.5)
        plt.plot(t_ref, step_fitted, label="Step-Fitted", linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (Hz)")
        plt.title("Step-Fitted Speed Trace")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.splitext(npz_path)[0] + "_speed_trace_stepfit.png"
        plt.savefig(out_path)
        print(f"Saved step-fitted speed plot to: {out_path}")
        plt.close()

    except Exception as e:
        print(f"Failed to analyze {npz_path}: {e}")

if __name__ == "__main__":
    path = "results_backup/2025.04.24 patricia/file/phase_data.npz"
    analyze_file(path)
