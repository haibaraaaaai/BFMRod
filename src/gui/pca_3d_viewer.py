from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout,
    QLineEdit, QComboBox
)
from PyQt6.QtGui import QShortcut, QKeySequence
import pyqtgraph.opengl as gl
import numpy as np

from processing.pca_module import apply_pca, detect_cycle_bounds, ref_cycle_update
from utils.smoothing import smooth_trajectory, smooth_data_with_convolution
from config import SAMPLING_RATE, DEFAULT_PCA_SEGMENT_DURATION, DEFAULT_CLOSURE_THRESHOLD, CONVOLUTION_WINDOW, FIRST_CYCLE_DETECTION_LIMIT, END_OF_CYCLE_LIMIT


class PCA3DViewer(QMainWindow):
    def __init__(self, data, timestamps):
        super().__init__()
        self.setWindowTitle("PCA 3D Viewer")
        self.setGeometry(200, 200, 1000, 700)

        self.data = data
        self.segment_duration = DEFAULT_PCA_SEGMENT_DURATION
        self.segment_size = int(self.segment_duration * SAMPLING_RATE)
        self.closure_threshold = DEFAULT_CLOSURE_THRESHOLD
        self.update_interval = 1.0
        self.alpha = 0.2
        self.fraction = 0.025

        pca, _ = apply_pca(self.data)
        self.pca = smooth_data_with_convolution(pca, CONVOLUTION_WINDOW)
        self.timestamps = timestamps[:len(self.pca)]

        ref_start, ref_end = detect_cycle_bounds(self.pca, self.closure_threshold)
        initial_cycle = self.pca[ref_start:ref_end]
        smooth_ref_cycle = smooth_trajectory(initial_cycle)

        init_segment_data = self.pca[:self.segment_size]
        init_seg_start_time = self.timestamps[0]
        init_seg_end_time = self.timestamps[self.segment_size - 1]
        self.pca_segments = [(init_segment_data, init_seg_start_time, init_seg_end_time)]
        self.computed_refs = [(ref_start, smooth_ref_cycle)]
        self.updated_refs = []
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

        self.min_time_label = QLabel(f"(Min: {self.timestamps[0]:.3f}s)")
        self.max_time_label = QLabel(f"(Max: {self.timestamps[-1]:.3f}s)")
        self.min_time_label.setStyleSheet("font-size: 10px; color: gray;")
        self.max_time_label.setStyleSheet("font-size: 10px; color: gray;")
        self.alpha_input = QLineEdit(str(self.alpha))
        self.fraction_input = QLineEdit(str(self.fraction))

        self.ref_selector = QComboBox()
        self.ref_selector.currentIndexChanged.connect(self.jump_to_ref_cycle)
        self.update_ref_selector()

        self.redo_closure_input = QLineEdit(str(self.closure_threshold))
        self.redo_button = QPushButton("Redo Selected Ref")
        self.redo_button.clicked.connect(self.redo_selected_ref)

        self.remove_button = QPushButton("Remove Selected Ref")
        self.remove_button.clicked.connect(self.remove_selected_ref)

        self.add_button = QPushButton("Add Ref Cycle")
        self.add_button.clicked.connect(self.add_ref_cycle)

        button_layout = QHBoxLayout()
        for widget in [
            QLabel("Start Time:"), self.start_time_input, self.min_time_label,
            QLabel("End Time:"), self.end_time_input, self.max_time_label,
            QLabel("Seg (s):"), self.segment_duration_input,
            QLabel("Update (s):"), self.update_interval_input,
            QLabel("Ref Cycles:"), self.ref_selector
        ]:
            button_layout.addWidget(widget)

        redo_layout = QHBoxLayout()
        for widget in [
            QLabel("Closure Threshold:"), self.redo_closure_input,
            self.redo_button, self.remove_button, self.add_button
        ]:
            redo_layout.addWidget(widget)

        fraction_alpha_layout = QHBoxLayout()
        for widget in [
            QLabel("Fraction:"), self.fraction_input,
            QLabel("Alpha:"), self.alpha_input
        ]:
            fraction_alpha_layout.addWidget(widget)

        self.recompute_button = QPushButton("Set PCA Window + Update Ref")
        self.recompute_button.clicked.connect(self.recompute_segments_and_ref)
        button_layout.addWidget(self.recompute_button)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(button_layout)
        layout.addLayout(fraction_alpha_layout)
        layout.addLayout(redo_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        QShortcut(QKeySequence(QtCore.Qt.Key.Key_Left), self).activated.connect(self.prev_segment)
        QShortcut(QKeySequence(QtCore.Qt.Key.Key_Right), self).activated.connect(self.next_segment)

    def update_ref_selector(self):
        self.ref_selector.clear()
        for i, (idx, _) in enumerate(self.computed_refs):
            timestamp = self.timestamps[idx]
            self.ref_selector.addItem(f"Ref {i+1} @ {timestamp:.3f}s")

    def jump_to_ref_cycle(self, index):
        if index < 0 or index >= len(self.computed_refs):
            return

        ref_start_idx, ref_cycle = self.computed_refs[index]
        ref_mid_idx = ref_start_idx + len(ref_cycle) // 2

        closest_segment_index = min(
            range(len(self.pca_segments)),
            key=lambda i: abs(
                np.searchsorted(self.timestamps, (self.pca_segments[i][1] + self.pca_segments[i][2]) / 2)
                - ref_mid_idx
            )
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

            self.all_refs = self.computed_refs + self.updated_refs
            segment_center_time = (seg_start_time + seg_end_time) / 2
            segment_center_idx = np.searchsorted(self.timestamps, segment_center_time)
            closest_ref_cycle = min(
                self.all_refs,
                key=lambda rc: abs(rc[0] - segment_center_idx)
            )[1] if self.all_refs else None

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
            if not self.computed_refs:
                QtWidgets.QMessageBox.warning(self, "Missing Ref", "Cannot update without any reference cycles.")
                return

            start_time = float(self.start_time_input.text())
            end_time = float(self.end_time_input.text())
            self.segment_duration = float(self.segment_duration_input.text())
            self.update_interval = float(self.update_interval_input.text())
            self.alpha = float(self.alpha_input.text())
            self.fraction = float(self.fraction_input.text())

            self.segment_size = int(self.segment_duration * SAMPLING_RATE)

            if start_time >= end_time:
                raise ValueError("Start time must be less than end time")

            pca_start_idx = np.searchsorted(self.timestamps, start_time, side='left')
            pca_end_idx = np.searchsorted(self.timestamps, end_time, side='right')
            windowed_pca = self.pca[pca_start_idx:pca_end_idx]

            updated_refs, _, _ = ref_cycle_update(
                windowed_pca, self.timestamps[pca_start_idx:pca_end_idx],
                self.computed_refs, pca_start_idx,
                update_interval=self.update_interval,
                fraction=self.fraction,
                alpha=self.alpha
            )
            self.updated_refs = updated_refs
            self.update_ref_selector()

            segments = []
            for i in range(0, len(windowed_pca), self.segment_size):
                seg = windowed_pca[i:i+self.segment_size]
                if len(seg) < self.segment_size:
                    break
                t0 = self.timestamps[pca_start_idx + i]
                t1 = self.timestamps[pca_start_idx + i + self.segment_size - 1]
                segments.append((seg, t0, t1))

            self.pca_segments = segments
            self.pca_index = 0
            if self.pca_segments:
                self.plot_pca_segment(self.pca_segments[0])

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Update Error", f"Failed to update PCA segments:\n{e}")

    def redo_selected_ref(self):
        try:
            ref_index = self.ref_selector.currentIndex()
            if ref_index < 0 or ref_index >= len(self.computed_refs):
                return

            new_closure_threshold = int(self.redo_closure_input.text())
            old_start_idx, _ = self.computed_refs[ref_index]
            redo_ref_start = max(0, old_start_idx - END_OF_CYCLE_LIMIT)
            redo_ref_end = redo_ref_start + FIRST_CYCLE_DETECTION_LIMIT + END_OF_CYCLE_LIMIT

            short_window = self.pca[redo_ref_start:redo_ref_end]
            local_start, local_end = detect_cycle_bounds(short_window, closure_threshold=new_closure_threshold)
            new_ref_start = redo_ref_start + local_start
            new_ref_end = redo_ref_start + local_end

            new_cycle = self.pca[new_ref_start:new_ref_end]
            smooth_ref = smooth_trajectory(new_cycle)

            self.computed_refs[ref_index] = (new_ref_start, smooth_ref)
            self.recompute_segments_and_ref()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ref Redo Error", f"Failed to redo ref cycle:\n{e}")

    def remove_selected_ref(self):
        ref_index = self.ref_selector.currentIndex()
        if 0 <= ref_index < len(self.computed_refs):
            self.computed_refs.pop(ref_index)
            self.updated_refs = []
            if not self.computed_refs:
                self.pca_segments = []
                self.update_ref_selector()
                self.view.clear()
            else:
                self.recompute_segments_and_ref()

    def add_ref_cycle(self):
        try:
            if self.computed_refs:
                seg_start_idx = np.searchsorted(self.timestamps, self.pca_segments[self.pca_index][1])
            else:
                seg_start_idx = 0

            seg_end_idx = seg_start_idx + FIRST_CYCLE_DETECTION_LIMIT + END_OF_CYCLE_LIMIT
            short_window = self.pca[seg_start_idx:seg_end_idx]
            local_start, local_end = detect_cycle_bounds(short_window, closure_threshold=self.closure_threshold)
            new_ref_start = seg_start_idx + local_start
            new_ref_end = seg_start_idx + local_end
            new_cycle = self.pca[new_ref_start:new_ref_end]
            smooth_ref = smooth_trajectory(new_cycle)
            self.computed_refs.append((new_ref_start, smooth_ref))
            self.recompute_segments_and_ref()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Add Ref Error", f"Failed to add new ref cycle:\n{e}")

    def prev_segment(self):
        if self.pca_index > 0:
            self.pca_index -= 1
            self.plot_pca_segment(self.pca_segments[self.pca_index])

    def next_segment(self):
        if self.pca_index < len(self.pca_segments) - 1:
            self.pca_index += 1
            self.plot_pca_segment(self.pca_segments[self.pca_index])
