"""
smoothing.py – Functions for smoothing signal data and 3D trajectories.
"""

import numpy as np
from scipy.interpolate import splprep, splev

from config import (
    CONVOLUTION_WINDOW,
    SMOOTHING,
    REFERENCE_NUM_POINTS,
)


def smooth_data_with_convolution(signals, window_size=CONVOLUTION_WINDOW):
    """
    Apply a moving average filter to smooth multi-channel signals.

    Args:
        signals (np.ndarray): Raw multi-channel signal data.
        window_size (int): Size of the moving average window.

    Returns:
        np.ndarray: Smoothed signal data (same shape as input, except reduced along time axis).
    """
    smoothed_signals = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(window_size) / window_size, mode="valid"),
        axis=0,
        arr=signals,
    )
    return smoothed_signals


def smooth_trajectory(points, smoothing_factor=SMOOTHING, num_points=REFERENCE_NUM_POINTS):
    """
    Apply cubic B-spline interpolation to smooth a closed 3D trajectory.

    Args:
        points (np.ndarray): Array of 3D points defining the trajectory.
        smoothing_factor (float): B-spline smoothing factor.
        num_points (int): Number of points in the smoothed trajectory.

    Returns:
        np.ndarray: Smoothed trajectory points with consistent spacing.
    """
    points = np.array(points)

    # Ensure the trajectory forms a closed loop
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    # Fit periodic B-spline
    tck, _ = splprep(points.T, per=True, s=smoothing_factor)
    u_fine = np.linspace(0, 1, num_points)
    smooth_points = np.array(splev(u_fine, tck)).T

    return smooth_points
