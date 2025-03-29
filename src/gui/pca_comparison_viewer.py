from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence
import pyqtgraph as pg
import numpy as np


class PCAComparisonViewer(QMainWindow):
    def __init__(self, timestamps, pca_trace, ref_trace, window_size=10000):
        super().__init__()
        self.setWindowTitle("PCA Comparison Viewer")
        self.setGeometry(150, 150, 1000, 600)

        self.timestamps = timestamps
        self.pca_trace = pca_trace
        self.ref_trace = ref_trace
        self.window_size = window_size
        self.min_window_size = 1000
        self.start_index = 0

        self._init_ui()
        self._init_shortcuts()
        self.update_plot()

    def _init_ui(self):
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)

        self.plot_widget = pg.PlotWidget(title="PCA0 vs Ref[Phase] Viewer")
        self.plot_widget.setLabel('left', "Signal")
        self.plot_widget.setLabel('bottom', "Time (s)")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setBackground('w')
        self.plot_widget.addLegend()

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.timestamps) - self.window_size)
        self.slider.setValue(self.start_index)
        self.slider.setSingleStep(self.window_size // 10)
        self.slider.valueChanged.connect(self.update_plot)

        self.layout.addWidget(self.plot_widget)
        self.layout.addWidget(self.slider)
        self.setCentralWidget(self.central_widget)

    def _init_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self.move_window_left)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self.move_window_right)
        QShortcut(QKeySequence(Qt.Key.Key_Up), self).activated.connect(self.increase_window_size)
        QShortcut(QKeySequence(Qt.Key.Key_Down), self).activated.connect(self.decrease_window_size)

    def update_plot(self):
        self.start_index = self.slider.value()
        end_index = min(self.start_index + self.window_size, len(self.timestamps))

        t_slice = self.timestamps[self.start_index:end_index]
        pca_slice = self.pca_trace[self.start_index:end_index]
        ref_slice = self.ref_trace[self.start_index:end_index]

        self.plot_widget.clear()
        self.plot_widget.plot(t_slice, pca_slice, pen=pg.mkPen('b', width=1), name="PCA0")
        self.plot_widget.plot(t_slice, ref_slice, pen=pg.mkPen('r', width=1), name="Ref[phase]")
        self.plot_widget.setXRange(t_slice[0], t_slice[-1])
        self.plot_widget.setYRange(float(min(np.min(pca_slice), np.min(ref_slice))),
                                   float(max(np.max(pca_slice), np.max(ref_slice))))

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
        self.update_plot()

    def decrease_window_size(self):
        step_size = max(100, int(self.window_size * 0.2))
        self.window_size = max(self.min_window_size, self.window_size - step_size)
        self.slider.setMaximum(len(self.timestamps) - self.window_size)
        self.update_plot()
