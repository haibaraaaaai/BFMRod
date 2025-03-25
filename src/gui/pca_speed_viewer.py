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
    Visualize instantaneous frequency derived from phase data using two methods:
        1. Gradient of phase (smoothed and decimated)
        2. Revolution-based frequency (cycle-to-cycle time)
    """

    def __init__(self, phase, phase_time, sampling_rate, decimation_factor=100, smoothing_window=51):
        super().__init__()

        self.setWindowTitle("Instantaneous Frequency Viewer")
        self.setGeometry(150, 150, 1000, 500)

        self.phase = phase
        self.time_full = phase_time
        self.dt = sampling_rate
        self.decimation_factor = decimation_factor
        self.smoothing_window = smoothing_window

        # --- Compute frequency and revolutions ---
        self.freq_gradient = self.compute_gradient_frequency()
        self.time_processed, self.freq_processed = self.process_frequency()
        self.rev_times, self.rev_freqs = self.compute_revolution_frequency(group_revs=3)

        # --- View Parameters ---
        self.window_size = 25_000
        self.start_index = 0
        self.min_window_size = 100

        # --- Plot Widget ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', "Frequency", units='Hz')
        self.plot_widget.setLabel('bottom', "Time", units='s')
        self.plot_widget.setYRange(0, 400)

        # Gradient-based trace (blue dashed line)
        self.plot_curve = self.plot_widget.plot(
            pen=pg.mkPen('b', width=1, style=Qt.PenStyle.DashLine)
        )

        # Revolution-based scatter (red dots)
        self.rev_scatter = self.plot_widget.plot(
            pen=None, symbol='o', symbolBrush='r', symbolSize=6
        )

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

        # --- Keyboard Shortcuts ---
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self.move_window_left)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self.move_window_right)
        QShortcut(QKeySequence(Qt.Key.Key_Up), self).activated.connect(self.increase_window_size)
        QShortcut(QKeySequence(Qt.Key.Key_Down), self).activated.connect(self.decrease_window_size)

        self.update_window()

    def compute_gradient_frequency(self):
        """
        Compute instantaneous frequency from the gradient of unwrapped phase.
        Returns:
            np.ndarray: Frequency in Hz.
        """
        speed = np.gradient(self.phase, self.dt)
        speed_smoothed = savgol_filter(speed, self.smoothing_window, polyorder=3)
        return speed_smoothed / (2 * np.pi)

    def process_frequency(self):
        """
        Decimate and clip the gradient-based frequency for visualization.
        Returns:
            (time_array, freq_array)
        """
        freq_dec = decimate(self.freq_gradient, self.decimation_factor, zero_phase=True)
        time_dec = self.time_full[::self.decimation_factor]
        freq_clipped = np.clip(freq_dec, 0, None)
        return time_dec, freq_clipped

    def compute_revolution_frequency(self, group_revs=3):
        """
        Compute frequency based on time between full 2π phase revolutions.

        Args:
            group_revs (int): Number of revolutions to average over.

        Returns:
            tuple of np.ndarray: (centered times, frequencies)
        """
        step = 2 * np.pi
        max_phase = self.phase[-1]
        thresholds = np.arange(step, max_phase, step)

        rev_times = []
        for threshold in thresholds:
            idx = np.searchsorted(self.phase, threshold)
            if idx >= len(self.time_full):
                break
            rev_times.append(self.time_full[idx])

        rev_times = np.array([self.time_full[0]] + rev_times)

        freq_times = []
        freq_values = []
        for i in range(len(rev_times) - group_revs):
            t_start = rev_times[i]
            t_end = rev_times[i + group_revs]
            freq = group_revs / (t_end - t_start)

            if group_revs % 2 == 0:
                mid = i + group_revs // 2
                freq_time = rev_times[mid]
            else:
                mid1 = i + (group_revs - 1) // 2
                mid2 = i + (group_revs + 1) // 2
                freq_time = (rev_times[mid1] + rev_times[mid2]) / 2

            freq_times.append(freq_time)
            freq_values.append(freq)

        return np.array(freq_times), np.array(freq_values)

    def update_window(self):
        """
        Update plot for current window slice, including frequency curve and revolution dots.
        """
        self.start_index = self.slider.value()
        end_index = min(self.start_index + self.window_size, len(self.time_processed))

        t_window = self.time_processed[self.start_index:end_index]
        f_window = self.freq_processed[self.start_index:end_index]
        self.plot_curve.setData(t_window, f_window)

        mask = (self.rev_times >= t_window[0]) & (self.rev_times <= t_window[-1])
        self.rev_scatter.setData(self.rev_times[mask], self.rev_freqs[mask])

        self.plot_widget.setXRange(t_window[0], t_window[-1], padding=0)

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
