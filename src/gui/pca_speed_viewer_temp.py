import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QSlider
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence
from scipy.signal import decimate, savgol_filter
from utils.filter_utils import chi2_filter_njit_slope_steps

class PCASpeedViewer(QMainWindow):
    def __init__(self, phase_data, phase_time, sampling_rate, decimation_factor=100, smoothing_window=51):
        super().__init__()

        self.setWindowTitle("Instantaneous Frequency Viewer")
        self.setGeometry(150, 150, 1000, 500)

        self.phase = phase_data
        self.time_full = phase_time
        self.sampling_rate = sampling_rate
        self.decimation_factor = decimation_factor
        self.smoothing_window = smoothing_window

        # --- Compute frequency ---
        self.time, self.freq_std, self.freq_chi2 = self.compute_frequency()
        self.time_processed, self.freq_std_processed, self.freq_chi2_processed = self.process_frequencies()

        # --- Parameters ---
        self.window_size = 25000
        self.start_index = 0
        self.window_step_fraction = 0.1
        self.min_window_size = 100

        # --- PlotWidget ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', "Frequency", units='Hz')
        self.plot_widget.setLabel('bottom', "Time", units='s')
        self.plot_widget.setYRange(0, 400)  # Fixed Y-axis range
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        self.curve_std = self.plot_widget.plot(pen='b', name='Standard Method')     # Blue
        self.curve_chi2 = self.plot_widget.plot(pen='r', name='Chi² Weighted Slope')  # Red

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

    def compute_frequency(self):
        dt = self.sampling_rate

        # --- Standard Method ---
        phase_smoothed = savgol_filter(self.phase, 51, polyorder=3)
        phase_unwrapped = np.unwrap(phase_smoothed)
        speed = np.gradient(phase_unwrapped, dt)
        speed_smoothed = savgol_filter(speed, 51, polyorder=3)
        freq_std = speed_smoothed / (2 * np.pi)

        # --- Chi2 Weighted Slope ---
        phase_unwrapped_chi2 = np.unwrap(self.phase)
        slope_filtered_chi2 = chi2_filter_njit_slope_steps(phase_unwrapped_chi2, sigma=0.05)
        freq_chi2 = slope_filtered_chi2 / (2 * np.pi)

        return self.time_full, freq_std, freq_chi2

    def process_frequencies(self):
        # --- Decimate and smooth standard method ---
        freq_std_dec = decimate(self.freq_std, self.decimation_factor, zero_phase=True)
        freq_std_smooth = savgol_filter(freq_std_dec, self.smoothing_window, polyorder=3)
        freq_std_smooth = np.clip(freq_std_smooth, 0, None)

        # --- Decimate and smooth chi² method ---
        freq_chi2_dec = decimate(self.freq_chi2, self.decimation_factor, zero_phase=True)
        freq_chi2_smooth = savgol_filter(freq_chi2_dec, self.smoothing_window, polyorder=3)
        freq_chi2_smooth = np.clip(freq_chi2_smooth, 0, None)

        # Time array downsampled
        time_dec = self.time[::self.decimation_factor]

        return time_dec, freq_std_smooth, freq_chi2_smooth

    def update_window(self):
        self.start_index = self.slider.value()
        end_index = min(self.start_index + self.window_size, len(self.time_processed))

        t_window = self.time_processed[self.start_index:end_index]
        f_std_window = self.freq_std_processed[self.start_index:end_index]
        f_chi2_window = self.freq_chi2_processed[self.start_index:end_index]

        self.curve_std.setData(t_window, f_std_window)
        self.curve_chi2.setData(t_window, f_chi2_window)
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