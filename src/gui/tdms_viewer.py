import os
import numpy as np
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QSlider, QCheckBox, QLineEdit, QPushButton, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence

from processing.tdms_loader import load_tdms_data
from utils.smoothing import smooth_data_with_convolution
from gui.pca_3d_viewer import PCA3DViewer

LAST_FILE_PATH = "last_file.txt"

class TDMSViewer(QMainWindow):
    def __init__(self, file_path=None):
        super().__init__()

        self.setGeometry(100, 100, 1000, 600)

        if file_path is None:
            file_path = self.prompt_for_file()
            if file_path is None:
                self.close()
                return

        self.last_file_path = file_path
        self.setWindowTitle(f"Viewing: {os.path.basename(self.last_file_path)}")
        self.load_data(self.last_file_path)

        self.window_size = 10_000
        self.start_index = 0
        self.min_window_size = 1_000

        self._init_layout()
        self.update_window()
        self._init_shortcuts()

    def prompt_for_file(self):
        initial_dir = ""
        if os.path.exists(LAST_FILE_PATH):
            with open(LAST_FILE_PATH, "r") as f:
                last_path = f.read().strip()
                if os.path.exists(last_path):
                    initial_dir = os.path.dirname(last_path)
        file_path, _ = QFileDialog.getOpenFileName(self, "Select TDMS File", initial_dir, "TDMS Files (*.tdms)")
        return file_path if file_path else None

    def load_data(self, file_path):
        self.timestamps, self.data, self.channel_names = load_tdms_data(file_path)
        if self.timestamps is None:
            self.close()
            return
        with open(LAST_FILE_PATH, "w") as f:
            f.write(file_path)
        self.smoothed_data = smooth_data_with_convolution(self.data)
        self.smoothed_timestamps = self.timestamps[:len(self.smoothed_data)]

    def _init_layout(self):
        self.central_widget = QWidget()
        self.final_layout = QVBoxLayout(self.central_widget)
        self.top_layout = QHBoxLayout()

        self.plot_widget = pg.PlotWidget(title="TDMS Data Viewer (Smoothed)")
        self.plot_widget.setLabel('left', "Signal")
        self.plot_widget.setLabel('bottom', "Time (s)")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setBackground('w')
        self.plot_widget.addLegend()
        self.top_layout.addWidget(self.plot_widget)
        self.series_list = []

        self.control_panel = QVBoxLayout()
        self.checkboxes = []
        for name in self.channel_names:
            checkbox = QCheckBox(name, self)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_window)
            self.control_panel.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        self.start_time_input = self._add_line_edit("Start Time (s)", 100)
        self.start_time_label = self._add_label_inline(f"Min: {self.smoothed_timestamps[0]:.3f}s")

        self.end_time_input = self._add_line_edit("End Time (s)", 100)
        self.end_time_label = self._add_label_inline(f"Max: {self.smoothed_timestamps[-1]:.3f}s")

        self.apply_button = QPushButton("Apply New Window")
        self.apply_button.clicked.connect(self.apply_manual_window)
        self.control_panel.addWidget(self.apply_button)

        self.pca_button = QPushButton("PCA 3D Viewer")
        self.pca_button.clicked.connect(self.launch_pca_viewer)
        self.control_panel.addWidget(self.pca_button)

        self.load_button = QPushButton("Load New TDMS File")
        self.load_button.clicked.connect(self.load_new_tdms_file)
        self.control_panel.addWidget(self.load_button)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red; font-size: 10px;")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.control_panel.addWidget(self.error_label)

        self.top_layout.addLayout(self.control_panel)
        self.final_layout.addLayout(self.top_layout)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.smoothed_timestamps) - self.window_size)
        self.slider.setValue(self.start_index)
        self.slider.setSingleStep(self.window_size // 10)
        self.slider.valueChanged.connect(self.update_window)
        self.final_layout.addWidget(self.slider)

        self.central_widget.setLayout(self.final_layout)
        self.setCentralWidget(self.central_widget)

        self.update_time_inputs_from_current_window()

    def _add_line_edit(self, label_text, width, default_text=""):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit()
        line_edit.setFixedWidth(width)
        if default_text:
            line_edit.setText(default_text)
        layout.addWidget(label)
        layout.addWidget(line_edit)
        self.control_panel.addLayout(layout)
        return line_edit

    def _add_label_inline(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-size: 10px;")
        label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.control_panel.addWidget(label)
        return label

    def _init_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self.move_window_left)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self.move_window_right)
        QShortcut(QKeySequence(Qt.Key.Key_Up), self).activated.connect(self.increase_window_size)
        QShortcut(QKeySequence(Qt.Key.Key_Down), self).activated.connect(self.decrease_window_size)

    def update_window(self):
        self.start_index = self.slider.value()
        end_index = min(self.start_index + self.window_size, len(self.smoothed_timestamps))

        time_window = self.smoothed_timestamps[self.start_index:end_index]
        selected_channels = [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]

        self.plot_widget.clear()
        self.series_list.clear()
        all_data = []

        for i in selected_channels:
            signal_slice = self.smoothed_data[self.start_index:end_index, i]
            time_slice = self.smoothed_timestamps[self.start_index:end_index]

            colors = ['r', 'g', 'b', 'm']
            pen = pg.mkPen(color=colors[i % len(colors)], width=1)
            plot_item = self.plot_widget.plot(time_slice, signal_slice, pen=pen, name=self.channel_names[i])
            self.series_list.append(plot_item)
            all_data.append(signal_slice)

        if all_data:
            flat_data = np.concatenate(all_data)
            self.plot_widget.setXRange(time_window[0], time_window[-1])
            self.plot_widget.setYRange(float(np.min(flat_data)), float(np.max(flat_data)))

        self.update_time_inputs_from_current_window()

    def update_time_inputs_from_current_window(self):
        start_t = self.smoothed_timestamps[self.start_index]
        end_index = min(self.start_index + self.window_size, len(self.smoothed_timestamps) - 1)
        end_t = self.smoothed_timestamps[end_index]
        self.start_time_input.setText(f"{start_t:.3f}")
        self.end_time_input.setText(f"{end_t:.3f}")

    def apply_manual_window(self):
        try:
            start_time = float(self.start_time_input.text())
            end_time = float(self.end_time_input.text())

            if start_time >= end_time:
                self.error_label.setText("⚠ Invalid time range.")
                return

            start_time = max(start_time, self.smoothed_timestamps[0])
            end_time = min(end_time, self.smoothed_timestamps[-1])

            start_index = np.searchsorted(self.smoothed_timestamps, start_time, side='left')
            end_index = np.searchsorted(self.smoothed_timestamps, end_time, side='right')

            self.start_index = start_index
            self.window_size = end_index - start_index
            self.slider.setValue(self.start_index)
            self.slider.setMaximum(len(self.smoothed_timestamps) - self.window_size)
            self.update_window()
            self.error_label.setText("")
        except ValueError:
            self.error_label.setText("⚠ Invalid number input.")

    def launch_pca_viewer(self):
        try:
            tdms_filename = os.path.splitext(os.path.basename(self.last_file_path))[0]
            parent_folder = os.path.basename(os.path.dirname(self.last_file_path))
            file_basename = os.path.join(parent_folder, tdms_filename)

            self.pca_viewer = PCA3DViewer(
                data=self.data,
                timestamps=self.timestamps,
                file_basename=file_basename
            )
            self.pca_viewer.show()
        except Exception as e:
            self.error_label.setText(f"⚠ PCA viewer error: {e}")

    def load_new_tdms_file(self):
        file_path = self.prompt_for_file()
        if file_path:
            self.last_file_path = file_path
            self.setWindowTitle(f"Viewing: {os.path.basename(self.last_file_path)}")
            self.load_data(file_path)
            self.start_index = 0
            self.window_size = 10_000
            self.slider.setValue(self.start_index)
            self.slider.setMaximum(len(self.smoothed_timestamps) - self.window_size)
            self.start_time_label.setText(f"Min: {self.smoothed_timestamps[0]:.3f}s")
            self.end_time_label.setText(f"Max: {self.smoothed_timestamps[-1]:.3f}s")
            self.plot_widget.clear()
            for checkbox in self.checkboxes:
                checkbox.setChecked(True)
            self.update_window()

    def move_window_left(self):
        step_size = max(100, int(self.window_size * 0.1))
        self.slider.setValue(max(0, self.start_index - step_size))

    def move_window_right(self):
        step_size = max(100, int(self.window_size * 0.1))
        self.slider.setValue(min(len(self.smoothed_timestamps) - self.window_size, self.start_index + step_size))

    def increase_window_size(self):
        step_size = max(100, int(self.window_size * 0.2))
        self.window_size = min(len(self.smoothed_timestamps), self.window_size + step_size)
        self.slider.setMaximum(len(self.smoothed_timestamps) - self.window_size)
        self.update_window()

    def decrease_window_size(self):
        step_size = max(100, int(self.window_size * 0.2))
        self.window_size = max(self.min_window_size, self.window_size - step_size)
        self.slider.setMaximum(len(self.smoothed_timestamps) - self.window_size)
        self.update_window()
