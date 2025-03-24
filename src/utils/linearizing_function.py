import numpy as np
from scipy.interpolate import CubicSpline
from scipy import stats
import matplotlib.pyplot as plt


def linearizing_speed_function(phir, speeds, N=500, fftfilter=1, mfilter=7, show=1):
    """
    Linearizes a speed function by correcting non-linearities using FFT filtering and cubic spline interpolation.
    Parameters:
    phir (array-like): Array of phase angles (in radians).
    speeds (array-like): Array of speed values corresponding to the phase angles.
    N (int, optional): Number of bins for histogram and binned statistics. Default is 500.
    fftfilter (int, optional): Flag to apply FFT filtering. Default is 1 (apply filtering).
    mfilter (int, optional): Number of FFT components to retain. Default is 7.
    show (int, optional): Flag to display plots. Default is 1 (show plots).
    Returns:
    function: A cubic spline function that maps the original phase angles to corrected phase angles.
    - If the `show` flag is set, the function generates plots to visualize the raw points, average signal, filtered signal, and the correction function.
    """

    nn, _ = np.histogram(np.array(phir) % (2 * np.pi), N)  # Compute the histo of m
    bins = np.linspace(0, 2 * np.pi, N)
    binned_std_displacement = stats.binned_statistic(phir, speeds, "std", bins=bins)
    sy = binned_std_displacement.statistic

    if show:
        plt.figure()
        plt.plot(
            np.array(m[1:]) % (2 * np.pi),
            speeds,
            ".",
            markersize=1,
            label="raw points",
        )
        plt.plot(np.linspace(0, 2 * np.pi, N), sy, label="Average signal")

    if 0 in nn:  # if undersampling can not correct non linearities.
        print("Warning : I did not correct non linearities because of undersampling")
        return lambda x: x

    # Here we FFT filter the mean
    if fftfilter:
        rft = np.fft.rfft(sy)
        rft[mfilter:] = 0  # Note, rft.shape = 21
        sy = np.fft.irfft(rft)
        if show:
            plt.plot(np.linspace(0, 2 * np.pi, N), sy, label="Filtered signal")
            plt.legend()
            plt.xlabel("Phir (rad)")
            plt.ylabel("Value")

    fcor = np.cumsum(1 / sy)
    fcor = np.insert(fcor, 0, 0)
    fcor = fcor * 2 * np.pi / fcor[-1]
    f = CubicSpline(np.linspace(0, 2 * np.pi, N - 1), fcor)

    if show:
        plt.figure()
        plt.plot(
            np.linspace(0, 2 * np.pi, N - 1), f(np.linspace(0, 2 * np.pi, N - 1)), ".-"
        )
        plt.xlabel("$\phi_{ori}$")
        plt.ylabel("$\phi_{old}$")

    return f
