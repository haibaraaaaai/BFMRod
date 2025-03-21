import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QCheckBox,
    QLineEdit, QPushButton
)
from PyQt6.QtCore import Qt
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPainter, QShortcut, QKeySequence

from scipy.signal import decimate
from config import SAMPLING_RATE
from processing import load_tdms_data, apply_pca
from utils import smooth_data_with_convolution
from gui.pca_3d_viewer import PCA3DViewer


class TDMSViewer(QMainWindow):
    def __init__(self, file_path):
        super().__init__()

        self.setWindowTitle(f"Viewing: {file_path.split('/')[-1]}")
        self.setGeometry(100, 100, 1000, 600)

        self.timestamps, self.data, self.channel_names = load_tdms_data(file_path)
        if self.timestamps is None:
            self.close()
            return

        # --- Parameters ---
        self.window_size = 10000
        self.start_index = 0
        self.decimation_factor = 100
        self.window_step_fraction = 0.1
        self.min_window_size = 1000

        # --- Layout ---
        self.central_widget = QWidget()
        self.final_layout = QVBoxLayout(self.central_widget)
        self.top_layout = QHBoxLayout()

        # --- Chart ---
        self.chart = QChart()
        self.chart.setTitle("TDMS Data Viewer")
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.top_layout.addWidget(self.chart_view)

        self.axis_x = QValueAxis()
        self.axis_y = QValueAxis()
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.series_list = []

        # --- Control Panel ---
        self.control_panel = QVBoxLayout()
        self.checkboxes = []
        for name in self.channel_names:
            checkbox = QCheckBox(name, self)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_window)
            self.control_panel.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        self.start_time_input = QLineEdit()
        self.start_time_input.setPlaceholderText("Start Time (s)")
        self.start_time_input.setFixedWidth(100)
        self.control_panel.addWidget(self.start_time_input)

        self.end_time_input = QLineEdit()
        self.end_time_input.setPlaceholderText("End Time (s)")
        self.end_time_input.setFixedWidth(100)
        self.control_panel.addWidget(self.end_time_input)

        self.time_range_label = QLabel(f"Min: {self.timestamps[0]:.3f}s / Max: {self.timestamps[-1]:.3f}s")
        self.control_panel.addWidget(self.time_range_label)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_manual_window)
        self.control_panel.addWidget(self.apply_button)

        # --- PCA Inputs ---
        self.pca_start_input = QLineEdit()
        self.pca_start_input.setPlaceholderText("PCA Start Time (s)")
        self.pca_start_input.setFixedWidth(120)
        self.pca_start_input.setText(f"{self.timestamps[0]:.3f}")
        self.control_panel.addWidget(self.pca_start_input)

        self.pca_end_input = QLineEdit()
        self.pca_end_input.setPlaceholderText("PCA End Time (s)")
        self.pca_end_input.setFixedWidth(120)
        self.pca_end_input.setText(f"{self.timestamps[-1]:.3f}")
        self.control_panel.addWidget(self.pca_end_input)

        self.segment_size_input = QLineEdit()
        self.segment_size_input.setPlaceholderText("Segment Size (s)")
        self.segment_size_input.setFixedWidth(120)
        self.segment_size_input.setText("0.01")
        self.control_panel.addWidget(self.segment_size_input)

        self.pca_button = QPushButton("Run PCA")
        self.pca_button.clicked.connect(self.run_pca_analysis)
        self.control_panel.addWidget(self.pca_button)

        # --- Error Display ---
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red; font-size: 10px;")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.control_panel.addWidget(self.error_label)

        self.top_layout.addLayout(self.control_panel)
        self.final_layout.addLayout(self.top_layout)

        # --- Slider ---
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.timestamps) - self.window_size)
        self.slider.setValue(self.start_index)
        self.slider.setSingleStep(self.window_size // 10)
        self.slider.valueChanged.connect(self.update_window)
        self.final_layout.addWidget(self.slider)

        self.central_widget.setLayout(self.final_layout)
        self.setCentralWidget(self.central_widget)

        self.update_window()

        # --- Keyboard Shortcuts ---
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self.move_window_left)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self.move_window_right)
        QShortcut(QKeySequence(Qt.Key.Key_Up), self).activated.connect(self.increase_window_size)
        QShortcut(QKeySequence(Qt.Key.Key_Down), self).activated.connect(self.decrease_window_size)

    def update_window(self):
        self.start_index = self.slider.value()
        end_index = min(self.start_index + self.window_size, len(self.timestamps))

        time_window = self.timestamps[self.start_index:end_index]
        selected_channels = [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]

        self.chart.removeAllSeries()
        self.series_list.clear()

        all_data = []
        for i in selected_channels:
            data_dec = decimate(self.data[self.start_index:end_index, i], self.decimation_factor)
            time_dec = np.linspace(time_window[0], time_window[-1], len(data_dec))

            series = QLineSeries()
            series.setUseOpenGL(True)
            for t, val in zip(time_dec, data_dec):
                series.append(t, val)

            series.setName(self.channel_names[i])
            self.chart.addSeries(series)
            series.attachAxis(self.axis_x)
            series.attachAxis(self.axis_y)
            self.series_list.append(series)
            all_data.append(data_dec)

        if all_data:
            flat_data = np.concatenate(all_data)
            self.axis_x.setRange(time_window[0], time_window[-1])
            self.axis_y.setRange(float(np.min(flat_data)), float(np.max(flat_data)))

        self.start_time_input.setText(f"{time_window[0]:.3f}")
        self.end_time_input.setText(f"{time_window[-1]:.3f}")

    def apply_manual_window(self):
        try:
            start_time = float(self.start_time_input.text())
            end_time = float(self.end_time_input.text())

            if start_time < self.timestamps[0] or end_time > self.timestamps[-1]:
                self.error_label.setText("⚠ Time out of range.")
                return

            start_index = np.searchsorted(self.timestamps, start_time)
            end_index = np.searchsorted(self.timestamps, end_time)

            if start_index >= end_index:
                self.error_label.setText("⚠ Invalid time range.")
                return

            self.start_index = start_index
            self.window_size = end_index - start_index
            self.slider.setValue(self.start_index)
            self.slider.setMaximum(len(self.timestamps) - self.window_size)
            self.update_window()
            self.error_label.setText("")
        except ValueError:
            self.error_label.setText("⚠ Invalid number input.")

    def run_pca_analysis(self):
        try:
            pca_start_time = float(self.pca_start_input.text())
            pca_end_time = float(self.pca_end_input.text())
            segment_duration = float(self.segment_size_input.text())

            if pca_start_time < self.timestamps[0] or pca_end_time > self.timestamps[-1]:
                self.error_label.setText("⚠ PCA time range out of bounds.")
                return

            segment_size = int(segment_duration * SAMPLING_RATE)
            start_idx = np.searchsorted(self.timestamps, pca_start_time)
            end_idx = np.searchsorted(self.timestamps, pca_end_time)

            if start_idx >= end_idx:
                self.error_label.setText("⚠ Invalid PCA time range.")
                return

            total_samples = end_idx - start_idx
            expected_segments = total_samples // segment_size

            if total_samples < segment_size:
                self.error_label.setText("⚠ Not enough data for PCA.")
                return

            raw_signals = self.data[start_idx:end_idx, :]
            smoothed = smooth_data_with_convolution(raw_signals)

            segments = []
            for i in range(expected_segments):
                seg_start = i * segment_size
                seg_end = seg_start + segment_size
                seg = smoothed[seg_start:seg_end]
                X_pca = apply_pca(seg)
                seg_start_time = self.timestamps[start_idx + seg_start]
                seg_end_time = self.timestamps[start_idx + seg_end - 1]
                segments.append((X_pca, seg_start_time, seg_end_time))

            if segments:
                self.error_label.setText("")
                self.pca_viewer = PCA3DViewer(segments)
                self.pca_viewer.show()

        except ValueError:
            self.error_label.setText("⚠ Invalid PCA input.")
        except Exception as e:
            self.error_label.setText(f"⚠ PCA error: {e}")

    # --- Navigation Shortcuts ---
    def move_window_left(self):
        step_size = int(self.window_size * self.window_step_fraction)
        self.slider.setValue(max(0, self.start_index - step_size))

    def move_window_right(self):
        step_size = int(self.window_size * self.window_step_fraction)
        self.slider.setValue(min(len(self.timestamps) - self.window_size, self.start_index + step_size))

    def increase_window_size(self):
        self.window_size = min(len(self.timestamps), self.window_size + self.min_window_size)
        self.slider.setMaximum(len(self.timestamps) - self.window_size)
        self.update_window()

    def decrease_window_size(self):
        self.window_size = max(self.min_window_size, self.window_size - self.min_window_size)
        self.slider.setMaximum(len(self.timestamps) - self.window_size)
        self.update_window()
