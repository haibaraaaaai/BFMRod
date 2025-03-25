"""
pca_speed_viewer.py – GUI for visualizing instantaneous frequency derived from unwrapped phase.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QSlider
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence
from scipy.signal import decimate, savgol_filter


class PCASpeedViewer(QMainWindow):
    """
    GUI for viewing instantaneous frequency over time from phase data.

    Features:
        - Computes frequency from phase derivative
        - Applies smoothing and decimation
        - Zoomable, scrollable plot with shortcuts
    """
    def __init__(self, phase, phase_time, sampling_rate, decimation_factor=100, smoothing_window=51):
        super().__init__()

        self.setWindowTitle("Instantaneous Frequency Viewer")
        self.setGeometry(150, 150, 1000, 500)

        self.time_full = phase_time
        self.dt = sampling_rate
        self.decimation_factor = decimation_factor
        self.smoothing_window = smoothing_window

        # --- Compute frequency ---
        self.frequency = self.compute_frequency(phase)
        self.time_processed, self.freq_processed = self.process_frequency()

        # --- Parameters ---
        self.window_size = 25_000
        self.start_index = 0
        self.min_window_size = 100

        # --- PlotWidget Setup ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', "Frequency", units='Hz')
        self.plot_widget.setLabel('bottom', "Time", units='s')
        self.plot_widget.setYRange(0, 400)
        self.plot_curve = self.plot_widget.plot(pen='b')

        # --- Layout ---
        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.time_processed) - self.window_size)
        self.slider.setValue(self.start_index)
        self.slider.setSingleStep(self.window_size // 10)
        self.slider.valueChanged.connect(self.update_window)
        layout.addWidget(self.slider)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # --- Shortcuts ---
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self.move_window_left)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self.move_window_right)
        QShortcut(QKeySequence(Qt.Key.Key_Up), self).activated.connect(self.increase_window_size)
        QShortcut(QKeySequence(Qt.Key.Key_Down), self).activated.connect(self.decrease_window_size)

        self.update_window()

    def compute_frequency(self, phase):
        """Compute instantaneous frequency from unwrapped phase."""
        speed = np.gradient(phase, self.dt)
        speed_smoothed = savgol_filter(speed, self.smoothing_window, polyorder=3)
        return speed_smoothed / (2 * np.pi)

    def process_frequency(self):
        """Decimate and smooth frequency for efficient plotting."""
        freq_dec = decimate(self.frequency, self.decimation_factor, zero_phase=True)
        time_dec = self.time_full[::self.decimation_factor]
        freq_clipped = np.clip(freq_dec, 0, None)
        return time_dec, freq_clipped

    def update_window(self):
        """Update visible plot window based on current slider value."""
        self.start_index = self.slider.value()
        end_index = min(self.start_index + self.window_size, len(self.time_processed))
        t_window = self.time_processed[self.start_index:end_index]
        f_window = self.freq_processed[self.start_index:end_index]
        self.plot_curve.setData(t_window, f_window)
        self.plot_widget.setXRange(t_window[0], t_window[-1], padding=0)

    # --- Navigation Shortcuts ---
    def move_window_left(self):
        step_size = max(100, int(self.window_size * 0.1))
        self.slider.setValue(max(0, self.start_index - step_size))

    def move_window_right(self):
        step_size = max(100, int(self.window_size * 0.1))
        self.slider.setValue(min(len(self.time_processed) - self.window_size, self.start_index + step_size))

    def increase_window_size(self):
        step_size = max(100, int(self.window_size * 0.2))
        self.window_size = min(len(self.time_processed), self.window_size + step_size)
        self.slider.setMaximum(len(self.time_processed) - self.window_size)
        self.update_window()

    def decrease_window_size(self):
        step_size = max(100, int(self.window_size * 0.2))
        self.window_size = max(self.min_window_size, self.window_size - step_size)
        self.slider.setMaximum(len(self.time_processed) - self.window_size)
        self.update_window()
