from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QCheckBox, 
    QLineEdit, QPushButton
)
from PyQt6.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import ticker
from scipy.signal import decimate
from processing.tdms_loader import load_tdms_data


class TDMSViewer(QMainWindow):
    def __init__(self, file_path):
        super().__init__()

        self.setWindowTitle(f"Viewing: {file_path.split('/')[-1]}")
        self.setGeometry(100, 100, 1000, 600)

        # Load TDMS file data
        self.file_path = file_path
        self.timestamps, self.data, self.channel_names = load_tdms_data(file_path)

        if self.timestamps is None:
            self.close()
            return

        # Default parameters
        self.window_size = 10000
        self.start_index = 0
        self.decimation_factor = 100
        self.window_step_fraction = 0.1  # Move 1/10th of window size
        self.min_window_size = 1000  # Smallest allowable window

        # --- Create Main Layout ---
        self.central_widget = QWidget()
        self.final_layout = QVBoxLayout(self.central_widget)

        # --- Create Top Section (Plot + Controls) ---
        self.top_layout = QHBoxLayout()

        # --- Create Plot Area ---
        self.figure, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.top_layout.addWidget(self.canvas)

        # --- Create Control Panel ---
        self.control_panel = QVBoxLayout()

        # ✅ Checkboxes for channel selection
        self.checkboxes = []
        for name in self.channel_names:
            checkbox = QCheckBox(name, self)
            checkbox.setChecked(True)  # Default: Checked (All channels displayed)
            checkbox.stateChanged.connect(self.update_window)
            self.control_panel.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        # ✅ Manual Input for Start/End Time
        self.start_time_layout = QHBoxLayout()

        self.start_time_input = QLineEdit()
        self.start_time_input.setPlaceholderText("Start Time (s)")
        self.start_time_input.setFixedWidth(100)
        self.start_time_input.setFocusPolicy(Qt.FocusPolicy.ClickFocus)  # Disable focus capture
        self.start_time_layout.addWidget(QLabel("Start:"))  # Add label
        self.start_time_layout.addWidget(self.start_time_input)

        self.start_time_min_label = QLabel(f"(Min: {self.timestamps[0]:.3f}s)")
        self.start_time_layout.addWidget(self.start_time_min_label)

        self.control_panel.addLayout(self.start_time_layout)

        self.end_time_layout = QHBoxLayout()

        self.end_time_input = QLineEdit()
        self.end_time_input.setPlaceholderText("End Time (s)")
        self.end_time_input.setFixedWidth(100)
        self.end_time_input.setFocusPolicy(Qt.FocusPolicy.ClickFocus)  # Disable focus capture
        self.end_time_layout.addWidget(QLabel("End:"))  # Add label
        self.end_time_layout.addWidget(self.end_time_input)

        self.end_time_max_label = QLabel(f"(Max: {self.timestamps[-1]:.3f}s)")
        self.end_time_layout.addWidget(self.end_time_max_label)

        self.control_panel.addLayout(self.end_time_layout)

        # ✅ Apply Button for Manual Input
        self.apply_button = QPushButton("Apply", self)
        self.apply_button.clicked.connect(self.apply_manual_window)
        self.control_panel.addWidget(self.apply_button)

        # --- PCA Analysis Section ---
        self.pca_layout = QHBoxLayout()

        self.pca_start_input = QLineEdit(self)
        self.pca_start_input.setPlaceholderText("PCA Start Time (s)")
        self.pca_start_input.setFixedWidth(120)
        self.pca_layout.addWidget(self.pca_start_input)

        self.pca_button = QPushButton("Run PCA", self)
        self.pca_button.clicked.connect(self.run_pca_analysis)
        self.pca_layout.addWidget(self.pca_button)

        self.final_layout.addLayout(self.pca_layout)

        # ✅ Tooltip for Error Messages
        self.error_label = QLabel("", self)
        self.error_label.setStyleSheet("color: red; font-size: 10px;")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.control_panel.addWidget(self.error_label)

        self.top_layout.addLayout(self.control_panel)
        self.final_layout.addLayout(self.top_layout)

        # --- Slider for Time Navigation ---
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.timestamps) - self.window_size)
        self.slider.setValue(self.start_index)
        self.slider.setSingleStep(self.window_size // 10)
        self.slider.valueChanged.connect(self.update_window)
        self.final_layout.addWidget(self.slider)

        self.central_widget.setLayout(self.final_layout)
        self.setCentralWidget(self.central_widget)

        self.update_window()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

    def update_window(self):
        """Updates the displayed TDMS data based on the current window position."""
        self.start_index = self.slider.value()
        end_index = min(self.start_index + self.window_size, len(self.timestamps))

        time_window = self.timestamps[self.start_index:end_index]
        selected_channels = [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]

        # Decimate and Plot
        self.ax.clear()
        for i in selected_channels:
            data_dec = decimate(self.data[self.start_index:end_index, i], self.decimation_factor)
            time_dec = np.linspace(time_window[0], time_window[-1], len(data_dec))
            self.ax.plot(time_dec, data_dec, label=self.channel_names[i])

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title("TDMS Data Viewer")

        if selected_channels:
            self.ax.legend()
        self.ax.grid(True)

        # ✅ Ensure Correct Formatting of Axis Labels
        self.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

        # ✅ Update Start/End Time Fields in GUI
        self.start_time_input.setText(f"{self.timestamps[self.start_index]:.3f}")
        self.end_time_input.setText(f"{self.timestamps[end_index-1]:.3f}")

        self.canvas.draw()

    def keyPressEvent(self, event):
        """Handles keyboard shortcuts for navigation and window resizing."""
        key = event.key()
        step_size = int(self.window_size * self.window_step_fraction)

        if key == Qt.Key.Key_Left:
            self.slider.setValue(max(0, self.start_index - step_size))

        elif key == Qt.Key.Key_Right:
            self.slider.setValue(min(len(self.timestamps) - self.window_size, self.start_index + step_size))

        elif key == Qt.Key.Key_Up:  # Increase window size
            self.window_size = min(len(self.timestamps), self.window_size + self.min_window_size)
            self.slider.setMaximum(len(self.timestamps) - self.window_size)
            self.update_window()

        elif key == Qt.Key.Key_Down:  # Decrease window size
            self.window_size = max(self.min_window_size, self.window_size - self.min_window_size)
            self.slider.setMaximum(len(self.timestamps) - self.window_size)
            self.update_window()

    def apply_manual_window(self):
        """Manually sets start and end time based on user input."""
        try:
            start_time = float(self.start_time_input.text())
            end_time = float(self.end_time_input.text())

            start_index = np.searchsorted(self.timestamps, start_time)
            end_index = np.searchsorted(self.timestamps, end_time)

            if start_index >= end_index or end_index > len(self.timestamps):
                self.error_label.setText("⚠ Invalid time range.")
                return

            self.start_index = start_index
            self.window_size = end_index - start_index
            self.slider.setValue(self.start_index)
            self.slider.setMaximum(len(self.timestamps) - self.window_size)

            self.update_window()
            self.error_label.setText("")
        except ValueError:
            self.error_label.setText("⚠ Enter valid numbers.")

    def run_pca_analysis(self):
        """Placeholder for running PCA analysis."""
        try:
            pca_start_time = float(self.pca_start_input.text())
            print(f"Starting PCA analysis from {pca_start_time}s")
        except ValueError:
            print("⚠ Invalid PCA start time entered.")
