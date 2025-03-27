"""
config.py – Centralized configuration constants for PCA, cycle detection, smoothing, and phase tracking.
"""

# ─── Data Acquisition ────────────────────────────────────────────────
SAMPLING_RATE = 250_000  # Hz – Samples per second

# ─── Smoothing Parameters ─────────────────────────────────────────────
CONVOLUTION_WINDOW = 100  # Moving average window size
SMOOTHING = 0.001         # B-spline smoothing factor

# ─── Cycle Detection ──────────────────────────────────────────────────
WINDOW_DETECTION = 400               # Sliding window size for periodicity detection
FIRST_CYCLE_DETECTION_LIMIT = 500_000  # Max search range for first cycle start
END_OF_CYCLE_LIMIT = 100_000         # Max search range for cycle end after start

# ─── Reference Cycle Settings ─────────────────────────────────────────
REFERENCE_NUM_POINTS = 200           # Number of interpolated points in reference cycle

# ─── Phase Tracking ───────────────────────────────────────────────────
CONTINUITY_CONSTRAINT = REFERENCE_NUM_POINTS // 10  # Neighborhood size for phase continuity search

# ─── For PCA 3D Viewer ───────────────────────────────────────────────────
DEFAULT_PCA_SEGMENT_DURATION = 1
DEFAULT_CLOSURE_THRESHOLD = 40