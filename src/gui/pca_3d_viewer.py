from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QShortcut, QKeySequence
import pyqtgraph.opengl as gl


class PCA3DViewer(QtWidgets.QMainWindow):
    def __init__(self, pca_segments, ref_cycles):
        super().__init__()

        self.setWindowTitle("3D PCA Trajectory Viewer")
        self.setGeometry(200, 200, 800, 600)

        self.pca_segments = pca_segments  # List of (X_pca, start_time, end_time)
        self.ref_cycles = ref_cycles      # List of (timestamp, ref_cycle)
        self.pca_index = 0

        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 10
        self.setCentralWidget(self.view)

        axis = gl.GLAxisItem()
        axis.setSize(x=3, y=3, z=3)
        self.view.addItem(axis)

        self.pca_line = None
        self.ref_line = None
        self.plot_pca_segment(self.pca_segments[self.pca_index])

        QShortcut(QKeySequence(QtCore.Qt.Key.Key_Left), self).activated.connect(self.prev_segment)
        QShortcut(QKeySequence(QtCore.Qt.Key.Key_Right), self).activated.connect(self.next_segment)

    def plot_pca_segment(self, segment_data):
        try:
            X_pca, seg_start_time, seg_end_time = segment_data

            # Safe removal of previous PCA line
            if self.pca_line and self.pca_line in self.view.items:
                self.view.removeItem(self.pca_line)

            # Safe removal of previous reference line
            if self.ref_line and self.ref_line in self.view.items:
                self.view.removeItem(self.ref_line)

            # Plot new PCA segment
            self.pca_line = gl.GLLinePlotItem(pos=X_pca, color=(0, 0, 1, 1), width=2.0, antialias=True)
            self.view.addItem(self.pca_line)

            # Find reference cycle with closest timestamp to segment center time
            segment_center_time = (seg_start_time + seg_end_time) / 2
            closest_ref_cycle = min(self.ref_cycles, key=lambda rc: abs(rc[0] - segment_center_time))[1] if self.ref_cycles else None

            if closest_ref_cycle is not None:
                self.ref_line = gl.GLLinePlotItem(pos=closest_ref_cycle, color=(1, 0, 0, 1), width=2.0, antialias=True)
                self.view.addItem(self.ref_line)

            self.setWindowTitle(
                f"3D PCA Viewer - Segment {self.pca_index + 1}/{len(self.pca_segments)} "
                f"[{seg_start_time:.3f}s – {seg_end_time:.3f}s]"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Plotting Error", f"Error plotting PCA segment:\n{e}")

    def prev_segment(self):
        if self.pca_index > 0:
            self.pca_index -= 1
            self.plot_pca_segment(self.pca_segments[self.pca_index])

    def next_segment(self):
        if self.pca_index < len(self.pca_segments) - 1:
            self.pca_index += 1
            self.plot_pca_segment(self.pca_segments[self.pca_index])
