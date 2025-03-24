# src/config/config.py

# Fixed Constants (Rarely Changed)
SAMPLING_RATE = 250000  # Hz (samples per second)
CONVOLUTION_WINDOW = 100  # Window size for moving average smoothing

# Cycle Detection Parameters
WINDOW_DETECTION = 400  # Sliding window size for periodicity detection
FIRST_CYCLE_DETECTION_LIMIT = 500000  # Search for cycle start within first N points
END_OF_CYCLE_LIMIT = 100000  # Search for cycle end within this range after start

# Reference Trajectory Settings
REFERENCE_NUM_POINTS = 200  # Number of points in the interpolated reference cycle
SMOOTHING = 0.001  # B-spline smoothing factor
DO_UPDATE_REFERENCE_CYCLE = True  # Enable drift correction
UPDATE_REFERENCE_CYCLE_SIZE = 250000  # Interval for updating reference cycle

# Phase Tracking
CONTINUITY_CONSTRAINT = REFERENCE_NUM_POINTS // 10  # Search neighborhood size for phase continuity, orignally // 10
