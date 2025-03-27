from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout,
    QLineEdit, QComboBox
)
from PyQt6.QtGui import QShortcut, QKeySequence
import pyqtgraph.opengl as gl
import numpy as np

from processing.pca_module import apply_pca, detect_cycle_bounds, ref_cycle_update
from utils.smoothing import smooth_trajectory
from config import SAMPLING_RATE, REFERENCE_NUM_POINTS, DEFAULT_PCA_SEGMENT_DURATION, DEFAULT_CLOSURE_THRESHOLD


class PCA3DViewer(QMainWindow):
    def __init__(self, data, timestamps):
        super().__init__()
        self.setWindowTitle("PCA 3D Viewer")
        self.setGeometry(200, 200, 1000, 700)

        self.data = data
        self.timestamps = timestamps
        self.segment_duration = DEFAULT_PCA_SEGMENT_DURATION
        self.segment_size = int(self.segment_duration * SAMPLING_RATE)
        self.closure_threshold = DEFAULT_CLOSURE_THRESHOLD
        self.update_interval = 1.0
        self.alpha = 0.2
        self.fraction = 0.025

        # Run PCA on full data
        self.pca, _ = apply_pca(self.data)

        # Build reference cycle
        ref_start, ref_end = detect_cycle_bounds(self.pca, self.closure_threshold)
        initial_cycle = self.pca[ref_start:ref_end]
        M = len(initial_cycle)
        avg_signal_d_av = np.zeros([M // REFERENCE_NUM_POINTS, 3])
        for i in range(M // REFERENCE_NUM_POINTS):
            avg_signal_d_av[i] = np.mean(initial_cycle[i * REFERENCE_NUM_POINTS : (i + 1) * REFERENCE_NUM_POINTS], axis=0)
        self.smooth_ref_cycle = smooth_trajectory(avg_signal_d_av)
        self.ref_start_idx = ref_start

        init_segment_data = self.pca[:self.segment_size]
        init_seg_start_time = self.timestamps[0]
        init_seg_end_time = self.timestamps[self.segment_size - 1]
        self.pca_segments = [(init_segment_data, init_seg_start_time, init_seg_end_time)]
        self.updated_refs = [(init_seg_start_time, self.smooth_ref_cycle)]
        self.detected_refs = [(init_seg_start_time, self.smooth_ref_cycle)]
        self.pca_index = 0

        self._init_ui()
        self.plot_pca_segment(self.pca_segments[0])

    def _init_ui(self):
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 10
        axis = gl.GLAxisItem()
        axis.setSize(x=3, y=3, z=3)
        self.view.addItem(axis)

        self.pca_line = None
        self.ref_line = None

        self.start_time_input = QLineEdit("0.0")
        self.end_time_input = QLineEdit(f"{self.timestamps[-1]:.3f}")
        self.segment_duration_input = QLineEdit(str(self.segment_duration))
        self.update_interval_input = QLineEdit(str(self.update_interval))

        # Labels for min/max
        self.min_time_label = QLabel(f"(Min: {self.timestamps[0]:.3f}s)")
        self.max_time_label = QLabel(f"(Max: {self.timestamps[-1]:.3f}s)")
        self.min_time_label.setStyleSheet("font-size: 10px; color: gray;")
        self.max_time_label.setStyleSheet("font-size: 10px; color: gray;")
        self.alpha_input = QLineEdit(str(self.alpha))
        self.fraction_input = QLineEdit(str(self.fraction))

        self.ref_selector = QComboBox()
        self.ref_selector.currentIndexChanged.connect(self.jump_to_ref_cycle)
        self.update_ref_selector()

        button_layout = QHBoxLayout()
        for widget in [
            QLabel("Start Time:"), self.start_time_input, self.min_time_label,
            QLabel("End Time:"), self.end_time_input, self.max_time_label,
            QLabel("Seg (s):"), self.segment_duration_input,
            QLabel("Update (s):"), self.update_interval_input,
            QLabel("Ref Cycles:"), self.ref_selector
        ]:
            button_layout.addWidget(widget)

        fraction_alpha_layout = QHBoxLayout()
        for widget in [
            QLabel("Fraction:"), self.fraction_input,
            QLabel("Alpha:"), self.alpha_input
        ]:
            fraction_alpha_layout.addWidget(widget)

        self.recompute_button = QPushButton("Update Segments + Ref")
        self.recompute_button.clicked.connect(self.recompute_segments_and_ref)
        button_layout.addWidget(self.recompute_button)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(button_layout)
        layout.addLayout(fraction_alpha_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        QShortcut(QKeySequence(QtCore.Qt.Key.Key_Left), self).activated.connect(self.prev_segment)
        QShortcut(QKeySequence(QtCore.Qt.Key.Key_Right), self).activated.connect(self.next_segment)

    def update_ref_selector(self):
        self.ref_selector.clear()
        all_refs = self.detected_refs + self.updated_refs
        for i, (t, _) in enumerate(all_refs):
            self.ref_selector.addItem(f"Ref {i+1} @ {t:.2f}s")

    def jump_to_ref_cycle(self, index):
        all_refs = self.detected_refs + self.updated_refs
        if index < 0 or index >= len(all_refs):
            return
        ref_time = all_refs[index][0]
        closest_segment_index = min(
            range(len(self.pca_segments)),
            key=lambda i: abs((self.pca_segments[i][1] + self.pca_segments[i][2]) / 2 - ref_time)
        )
        self.pca_index = closest_segment_index
        self.plot_pca_segment(self.pca_segments[self.pca_index])

    def plot_pca_segment(self, segment_data):
        try:
            X_pca, seg_start_time, seg_end_time = segment_data

            if self.pca_line and self.pca_line in self.view.items:
                self.view.removeItem(self.pca_line)
            if self.ref_line and self.ref_line in self.view.items:
                self.view.removeItem(self.ref_line)

            self.pca_line = gl.GLLinePlotItem(pos=X_pca, color=(0, 0, 1, 1), width=2.0, antialias=True)
            self.view.addItem(self.pca_line)

            segment_center_time = (seg_start_time + seg_end_time) / 2
            all_refs = self.detected_refs + self.updated_refs
            closest_ref_cycle = min(
                all_refs,
                key=lambda rc: abs(rc[0] - segment_center_time)
            )[1] if all_refs else None

            if closest_ref_cycle is not None:
                self.ref_line = gl.GLLinePlotItem(pos=closest_ref_cycle, color=(1, 0, 0, 1), width=2.0, antialias=True)
                self.view.addItem(self.ref_line)

            self.setWindowTitle(
                f"3D PCA Viewer - Segment {self.pca_index + 1}/{len(self.pca_segments)} "
                f"[{seg_start_time:.3f}s – {seg_end_time:.3f}s]"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Plotting Error", f"Error plotting PCA segment:\n{e}")

    def recompute_segments_and_ref(self):
        try:
            start_time = float(self.start_time_input.text())
            end_time = float(self.end_time_input.text())
            self.segment_duration = float(self.segment_duration_input.text())
            self.update_interval = float(self.update_interval_input.text())
            self.alpha = float(self.alpha_input.text())
            self.fraction = float(self.fraction_input.text())

            self.segment_size = int(self.segment_duration * SAMPLING_RATE)

            if start_time >= end_time:
                raise ValueError("Start time must be less than end time")

            start_idx = np.searchsorted(self.timestamps, start_time, side='left')
            end_idx = np.searchsorted(self.timestamps, end_time, side='right')
            pca_window = self.pca[start_idx:end_idx]

            updated_refs, _, _ = ref_cycle_update(
                pca_window, self.timestamps[start_idx:end_idx], self.smooth_ref_cycle,
                start_idx, self.ref_start_idx - start_idx,
                update_interval=self.update_interval,
                fraction=self.fraction,
                alpha=self.alpha
            )
            self.updated_refs = updated_refs
            self.update_ref_selector()

            segments = []
            for i in range(0, len(pca_window), self.segment_size):
                seg = pca_window[i:i+self.segment_size]
                if len(seg) < self.segment_size:
                    break
                t0 = self.timestamps[start_idx + i]
                t1 = self.timestamps[start_idx + i + self.segment_size - 1]
                segments.append((seg, t0, t1))

            self.pca_segments = segments
            self.pca_index = 0
            if self.pca_segments:
                self.plot_pca_segment(self.pca_segments[0])

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Update Error", f"Failed to update PCA segments:\n{e}")

    def prev_segment(self):
        if self.pca_index > 0:
            self.pca_index -= 1
            self.plot_pca_segment(self.pca_segments[self.pca_index])

    def next_segment(self):
        if self.pca_index < len(self.pca_segments) - 1:
            self.pca_index += 1
            self.plot_pca_segment(self.pca_segments[self.pca_index])