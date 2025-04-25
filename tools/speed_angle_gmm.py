"""
Analyze rotational speed distributions at a fixed angular window using GMM fitting.
Loads phase data, extracts per-revolution speeds, and fits Gaussian mixture models.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# Settings
REFERENCE_NUM_POINTS = 200
WINDOW_WIDTH_RAD = 0.2 * np.pi
WINDOW_STRIDE_RAD = 0.05 * np.pi
SMOOTH_WINDOW = 51
SMOOTH_POLYORDER = 3
LIMIT_SAMPLES = 1_500_000
MIN_POINTS = 10
MAX_SPEED_HZ = 600

npz_path = "results_backup/2025.04.15 patricia/file/phase_data.npz"


def compute_speed_vector(phase, phase_time, angle_center_rad, angle_width_rad=0.4, min_points=10):
    """Compute a per-revolution speed vector at a fixed angular window."""
    rev_count = int(phase[-1] // (2 * np.pi))
    speed_vector = []
    fit_data = []

    for rev in range(rev_count):
        rev_offset = rev * 2 * np.pi
        low = rev_offset + angle_center_rad - angle_width_rad / 2
        high = rev_offset + angle_center_rad + angle_width_rad / 2
        mask = (phase >= low) & (phase < high)
        if np.count_nonzero(mask) >= min_points:
            x = phase_time[mask]
            y = phase[mask]
            A = np.vstack([x, np.ones_like(x)]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            speed = slope / (2 * np.pi)
            speed_vector.append(speed)
            fit_data.append((x, y, slope))

    return np.array(speed_vector), fit_data


from sklearn.mixture import GaussianMixture

def analyze_file_histograms(npz_path):
    # Load and unwrap phase
    data = np.load(npz_path)
    phase0 = data["phase0"][:LIMIT_SAMPLES]
    phase_time = data["phase_time"][:LIMIT_SAMPLES]
    base = os.path.splitext(npz_path)[0]

    phase = phase0 / REFERENCE_NUM_POINTS * 2 * np.pi
    phase = np.unwrap(phase)
    phase = savgol_filter(phase, SMOOTH_WINDOW, SMOOTH_POLYORDER)

    # Define angular window
    ANGLE_CENTER_RAD = 2 * np.pi

    # Compute speed vector
    speeds, fit_data = compute_speed_vector(
        phase,
        phase_time,
        angle_center_rad=ANGLE_CENTER_RAD,
        angle_width_rad=WINDOW_WIDTH_RAD,
        min_points=MIN_POINTS,
    )
    speeds = speeds[np.isfinite(speeds) & (speeds <= MAX_SPEED_HZ)]

    if len(speeds) == 0:
        print("No valid speeds found for the specified angular window.")
        return

    # Plot phase and fits
    for i, (x, y, slope) in enumerate(fit_data[:10]):
        plt.figure()
        plt.scatter(x, y, s=10, alpha=0.6, label="Phase data")
        intercept = np.mean(y - slope * x)
        plt.plot(x, slope * x + intercept, 'r-', label="Fitted line")
        plt.xlabel("Time")
        plt.ylabel("Phase")
        plt.title(f"Phase and Fit for Revolution {i+1}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(base + f"_fit_phase_rev_{i+1}.png")
        plt.close()

    # Fit GMM using AIC
    bic_scores = []
    aic_scores = []
    models = []
    speeds_reshaped = speeds.reshape(-1, 1)
    for n in range(1, 7):
        gmm = GaussianMixture(n_components=n, reg_covar=1e-3, tol=1e-5)
        gmm.fit(speeds_reshaped)
        bic_scores.append(gmm.bic(speeds_reshaped))
        aic_scores.append(gmm.aic(speeds_reshaped))
        models.append(gmm)

    best_idx = np.argmin(aic_scores)
    best_gmm = models[best_idx]
    n_best = best_idx + 1

    # Plot histogram and GMM
    plt.figure()
    # Histogram
    plt.hist(speeds, bins=100, color='steelblue', alpha=0.6, density=True, label="Histogram")

    x_vals = np.linspace(0, MAX_SPEED_HZ, 500)

    # Plot individual Gaussian components
    means = best_gmm.means_.flatten()
    stds = np.sqrt(best_gmm.covariances_).flatten()
    weights = best_gmm.weights_.flatten()
    for i in range(n_best):
        component = weights[i] * (1/(stds[i]*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_vals-means[i])/stds[i])**2)
        plt.plot(x_vals, component, '--', linewidth=1.5, label=f"Gaussian {i+1}")

    plt.xlabel("Speed (Hz)")
    plt.ylabel("Density")
    plt.title(f"Speed Histogram & GMM at Angle {ANGLE_CENTER_RAD / np.pi:.2f}π rad\nBest n={n_best} (AIC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base + f"_hist_gmm_angle_{ANGLE_CENTER_RAD/np.pi:.2f}pi.png")
    plt.close()

    print(f"Histogram with GMM (n={n_best}) saved for angle {ANGLE_CENTER_RAD/np.pi:.2f}π rad.")


if __name__ == "__main__":
    analyze_file_histograms(npz_path)