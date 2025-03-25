"""
phase_viewer.py – GUI for visualizing unwrapped phase over time.
"""

import numpy as np
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QSlider
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence
import pyqtgraph as pg


class PhaseViewer(QMainWindow):
    """
    GUI for viewing phase evolution over time.

    Features:
        - Scrollable, zoomable phase plot
        - Keyboard shortcuts for navigation
    """
    def __init__(self, time_vector, phase_vector):
        super().__init__()

        self.setWindowTitle("Phase vs Time Viewer")
        self.setGeometry(150, 150, 1000, 500)

        self.time = time_vector
        self.phase = phase_vector

        # --- Parameters ---
        self.window_size = 10_000
        self.start_index = 0
        self.min_window_size = 1_000

        # --- Layout & Plot Setup ---
        self._init_layout()
        self._init_shortcuts()
        self.update_window()

    def _init_layout(self):
        self.plot_widget = pg.PlotWidget(title="Phase (radians) vs Time (s)")
        self.plot_widget.setLabel('left', "Phase (rad)")
        self.plot_widget.setLabel('bottom', "Time (s)")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setBackground('w')

        self.curve = self.plot_widget.plot(pen=pg.mkPen(color='r', width=1), name='Phase')

        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.time) - self.window_size)
        self.slider.setValue(self.start_index)
        self.slider.setSingleStep(self.window_size // 10)
        self.slider.valueChanged.connect(self.update_window)
        layout.addWidget(self.slider)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _init_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self.move_window_left)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self.move_window_right)
        QShortcut(QKeySequence(Qt.Key.Key_Up), self).activated.connect(self.increase_window_size)
        QShortcut(QKeySequence(Qt.Key.Key_Down), self).activated.connect(self.decrease_window_size)

    def update_window(self):
        self.start_index = self.slider.value()
        end_index = min(self.start_index + self.window_size, len(self.time))

        t_window = self.time[self.start_index:end_index]
        p_window = self.phase[self.start_index:end_index]

        self.curve.setData(t_window, p_window)
        self.plot_widget.setXRange(t_window[0], t_window[-1])
        self.plot_widget.setYRange(np.min(p_window), np.max(p_window))

    def move_window_left(self):
        step_size = max(100, int(self.window_size * 0.1))
        self.slider.setValue(max(0, self.start_index - step_size))

    def move_window_right(self):
        step_size = max(100, int(self.window_size * 0.1))
        self.slider.setValue(min(len(self.time) - self.window_size, self.start_index + step_size))

    def increase_window_size(self):
        step_size = max(100, int(self.window_size * 0.2))
        self.window_size = min(len(self.time), self.window_size + step_size)
        self.slider.setMaximum(len(self.time) - self.window_size)
        self.update_window()

    def decrease_window_size(self):
        step_size = max(100, int(self.window_size * 0.2))
        self.window_size = max(self.min_window_size, self.window_size - step_size)
        self.slider.setMaximum(len(self.time) - self.window_size)
        self.update_window()