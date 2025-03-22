import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QCheckBox,
    QLineEdit, QPushButton
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence

import pyqtgraph as pg
from scipy.signal import decimate
from config import SAMPLING_RATE, CONVOLUTION_WINDOW, REFERENCE_NUM_POINTS
from processing import load_tdms_data, apply_pca, detect_cycle_bounds, assign_phase_indices
from utils import smooth_data_with_convolution, smooth_trajectory
from gui.pca_3d_viewer import PCA3DViewer
from gui.phase_viewer import PhaseViewer
from gui.pca_speed_viewer import PCASpeedViewer


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

        # --- Plot Widget (PyQtGraph) ---
        self.plot_widget = pg.PlotWidget(title="TDMS Data Viewer")
        self.plot_widget.setLabel('left', "Signal")
        self.plot_widget.setLabel('bottom', "Time (s)")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setBackground('w')  # White background (optional)
        self.plot_widget.addLegend()
        self.top_layout.addWidget(self.plot_widget)

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
        self.segment_size_input.setText("1")
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

        self.plot_widget.clear()
        self.series_list.clear()
        all_data = []

        for i in selected_channels:
            data_dec = decimate(self.data[self.start_index:end_index, i], self.decimation_factor)
            time_dec = np.linspace(time_window[0], time_window[-1], len(data_dec))

            colors = ['r', 'g', 'b', 'm', 'c', 'y']
            pen = pg.mkPen(color=colors[i % len(colors)], width=1)
            plot_item = self.plot_widget.plot(time_dec, data_dec, pen=pen, name=self.channel_names[i])
            self.series_list.append(plot_item)
            all_data.append(data_dec)

        if all_data:
            flat_data = np.concatenate(all_data)
            self.plot_widget.setXRange(time_window[0], time_window[-1])
            self.plot_widget.setYRange(float(np.min(flat_data)), float(np.max(flat_data)))

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

            # --- Calculate segment info ---
            segment_size = int(segment_duration * SAMPLING_RATE)
            total_time = pca_end_time - pca_start_time
            expected_segments = int(total_time / segment_duration)

            total_samples_needed = expected_segments * segment_size
            pad_samples = CONVOLUTION_WINDOW - 1
            raw_end_sample = total_samples_needed + pad_samples

            start_idx = np.searchsorted(self.timestamps, pca_start_time)
            end_idx = min(start_idx + raw_end_sample, len(self.timestamps))  # Safety cap


            # --- Fetch Raw Data ---
            raw_signals = self.data[start_idx:end_idx, :]

            # --- Smoothing ---
            smoothed_signals = smooth_data_with_convolution(raw_signals)

            # --- PCA ---
            X_pca = apply_pca(smoothed_signals)

            # --- Segment PCA Data (no additional smoothing) ---
            pca_segments = []
            for seg_idx in range(expected_segments):
                seg_start = seg_idx * segment_size
                seg_end = seg_start + segment_size
                if seg_end > X_pca.shape[0]:
                    break
                segment_data = X_pca[seg_start:seg_end]
                seg_start_time = self.timestamps[start_idx + pad_samples + seg_start]
                seg_end_time = self.timestamps[start_idx + pad_samples + seg_end - 1]
                pca_segments.append((segment_data, seg_start_time, seg_end_time))


            # --- Reference Cycle ---
            ref_start, ref_end = detect_cycle_bounds(X_pca)

            initial_cycle = X_pca[ref_start:ref_end]
            M = len(initial_cycle)
            avg_signal_d_av = np.zeros([M // REFERENCE_NUM_POINTS, 3])
            for i in range(M // REFERENCE_NUM_POINTS):
                avg_signal_d_av[i] = np.mean(initial_cycle[i * REFERENCE_NUM_POINTS : (i + 1) * REFERENCE_NUM_POINTS], axis=0)
            smooth_ref_cycle = smooth_trajectory(avg_signal_d_av)

            # --- Update Ref Cycle Averaging Last Fixed Window ---
            update_interval_samples = SAMPLING_RATE
            total_samples_pca = X_pca.shape[0]
            updated_refs = [(self.timestamps[start_idx + ref_start], smooth_ref_cycle)]

            # Initialize phase tracking
            i = 0
            phase0 = np.array([], dtype=np.int32)
            prev_phase = None

            while i < total_samples_pca:
                # Get last 1 second of data
                segment = X_pca[i : min(i + update_interval_samples, total_samples_pca)]

                if segment.shape[0] == 0:
                    break

                phase_indices = assign_phase_indices(segment, smooth_ref_cycle, prev_phase)

                # Create new ref by averaging per phase
                phase_bins = [[] for _ in range(len(smooth_ref_cycle))]
                fraction = 0.05
                cut_start = int(len(segment) * (1 - fraction))
                for idx in range(cut_start, len(segment)):
                    phase_bins[phase_indices[idx]].append(segment[idx])

                # Interpolate missing phases
                if all(phase_bins):
                    pass  # No interpolation needed
                else:
                    for p_idx in range(len(phase_bins)):
                        if not phase_bins[p_idx]:
                            left = (p_idx - 1) % len(smooth_ref_cycle)
                            right = (p_idx + 1) % len(smooth_ref_cycle)
                            # Find nearest non-empty bins
                            while not phase_bins[left]:
                                left = (left - 1) % len(smooth_ref_cycle)
                            while not phase_bins[right]:
                                right = (right + 1) % len(smooth_ref_cycle)
                            # Average left and right
                            interpolated = 0.5 * (np.mean(phase_bins[left], axis=0) + np.mean(phase_bins[right], axis=0))
                            phase_bins[p_idx] = [interpolated]

                new_ref = np.array([np.median(points, axis=0) for points in phase_bins])
                ALPHA = 0.2  # Blend factor
                blended_ref = (1 - ALPHA) * smooth_ref_cycle + ALPHA * new_ref
                smooth_ref_cycle = smooth_trajectory(blended_ref)
                updated_refs.append((self.timestamps[start_idx + i], smooth_ref_cycle))

                i += update_interval_samples
                phase0 = np.concatenate((phase0, phase_indices))
                prev_phase = phase_indices[-1]

            phase = phase0 / len(smooth_ref_cycle) * 2 * np.pi
            phase = np.unwrap(phase)

            self.phase_data = phase
            self.phase_time = self.timestamps[start_idx : start_idx + len(phase)]

            if pca_segments:
                self.pca_viewer = PCA3DViewer(pca_segments, updated_refs)
                self.pca_viewer.show()

                self.phase_viewer = PhaseViewer(self.phase_time, self.phase_data)
                self.phase_viewer.show()

            try:
                self.speed_viewer = PCASpeedViewer(self.phase_data, self.phase_time, sampling_rate=1 / SAMPLING_RATE)
                self.speed_viewer.show()
            except Exception as e:
                self.error_label.setText(f"⚠ PCA error: {e}")

        except ValueError:
            self.error_label.setText("⚠ Invalid PCA input.")
        except Exception as e:
            self.error_label.setText(f"⚠ PCA error: {e}")

    # --- Navigation Shortcuts ---
    def move_window_left(self):
        step_size = max(100, int(self.window_size * 0.1))
        self.slider.setValue(max(0, self.start_index - step_size))

    def move_window_right(self):
        step_size = max(100, int(self.window_size * 0.1))
        self.slider.setValue(min(len(self.timestamps) - self.window_size, self.start_index + step_size))

    def increase_window_size(self):
        step_size = max(100, int(self.window_size * 0.2))
        self.window_size = min(len(self.timestamps), self.window_size + step_size)
        self.slider.setMaximum(len(self.timestamps) - self.window_size)
        self.update_window()

    def decrease_window_size(self):
        step_size = max(100, int(self.window_size * 0.2))
        self.window_size = max(self.min_window_size, self.window_size - step_size)
        self.slider.setMaximum(len(self.timestamps) - self.window_size)
        self.update_window()
