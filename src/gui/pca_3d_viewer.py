import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QShortcut, QKeySequence
import pyqtgraph.opengl as gl
import pyqtgraph as pg


class PCA3DViewer(QtWidgets.QMainWindow):
    def __init__(self, pca_segments):
        super().__init__()

        self.setWindowTitle("3D PCA Trajectory Viewer")
        self.setGeometry(200, 200, 800, 600)

        self.pca_segments = pca_segments  # List of (X_pca, start_time, end_time)
        self.pca_index = 0

        # --- Central Widget: OpenGL 3D View ---
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 10
        self.setCentralWidget(self.view)

        # --- Add 3D Axes ---
        axis = gl.GLAxisItem()
        axis.setSize(x=3, y=3, z=3)
        self.view.addItem(axis)

        # --- Current Line Plot ---
        self.pca_line = None
        self.plot_pca_segment(self.pca_segments[self.pca_index])

        # --- Arrow Key Navigation ---
        QShortcut(QKeySequence(QtCore.Qt.Key.Key_Left), self).activated.connect(self.prev_segment)
        QShortcut(QKeySequence(QtCore.Qt.Key.Key_Right), self).activated.connect(self.next_segment)

    def plot_pca_segment(self, segment_data):
        try:
            X_pca, seg_start_time, seg_end_time = segment_data

            if self.pca_line:
                self.view.removeItem(self.pca_line)

            # Create line plot in 3D
            line = gl.GLLinePlotItem(pos=X_pca, color=(0, 0, 1, 1), width=2.0, antialias=True)
            self.view.addItem(line)
            self.pca_line = line

            # Update window title with time info
            self.setWindowTitle(
                f"3D PCA Viewer - Segment {self.pca_index + 1}/{len(self.pca_segments)} "
                f"[{seg_start_time:.3f}s – {seg_end_time:.3f}s]"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Plotting Error", f"An error occurred while plotting PCA segment:\n{e}"
            )

    def prev_segment(self):
        if self.pca_index > 0:
            self.pca_index -= 1
            self.plot_pca_segment(self.pca_segments[self.pca_index])

    def next_segment(self):
        if self.pca_index < len(self.pca_segments) - 1:
            self.pca_index += 1
            self.plot_pca_segment(self.pca_segments[self.pca_index])
