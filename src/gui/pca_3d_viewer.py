from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout,
    QLineEdit, QComboBox
)
from PyQt6.QtGui import QShortcut, QKeySequence
import pyqtgraph.opengl as gl
import numpy as np
from scipy.signal import savgol_filter
import os
import json

from processing.pca_module import apply_pca, detect_cycle_bounds, ref_cycle_update
from utils.smoothing import smooth_trajectory, smooth_data_with_convolution
from config import SAMPLING_RATE, DEFAULT_PCA_SEGMENT_DURATION, DEFAULT_CLOSURE_THRESHOLD, CONVOLUTION_WINDOW, FIRST_CYCLE_DETECTION_LIMIT, END_OF_CYCLE_LIMIT, REFERENCE_NUM_POINTS
from gui.pca_comparison_viewer import PCAComparisonViewer
from gui.phase_viewer import PhaseViewer
from gui.pca_speed_viewer import PCASpeedViewer
from cache.ref_update_cache import RefUpdateCacheManager, SegmentCacheEntry


class PCA3DViewer(QMainWindow):
    def __init__(self, data, timestamps, file_basename="unnamed"):
        super().__init__()
        self.setWindowTitle("PCA 3D Viewer")
        self.setGeometry(200, 200, 1000, 700)

        self.file_basename = file_basename
        self.segment_duration = DEFAULT_PCA_SEGMENT_DURATION
        self.segment_size = int(self.segment_duration * SAMPLING_RATE)
        self.closure_threshold = DEFAULT_CLOSURE_THRESHOLD
        self.update_interval = 1.0
        self.alpha = 0.2
        self.fraction = 0.025

        pca, _ = apply_pca(data)
        savgol_pca = savgol_filter(pca, window_length=CONVOLUTION_WINDOW, polyorder=3, axis=0)
        self.pca = smooth_data_with_convolution(savgol_pca, CONVOLUTION_WINDOW)
        self.timestamps = timestamps[:len(self.pca)]

        ref_start, ref_end = detect_cycle_bounds(self.pca, self.closure_threshold)
        initial_cycle = self.pca[ref_start:ref_end]
        smooth_ref_cycle = smooth_trajectory(initial_cycle)

        init_segment_data = self.pca[:self.segment_size]
        init_seg_start_time = self.timestamps[0]
        init_seg_end_time = self.timestamps[self.segment_size - 1]
        self.pca_segments = [(init_segment_data, init_seg_start_time, init_seg_end_time)]
        self.computed_refs = [(ref_start, smooth_ref_cycle)]
        self.computed_refs_bound = [(ref_start, ref_end)]
        self.valid_refs = list(self.computed_refs)
        self.all_valid_refs = list(self.computed_refs)
        self.pca_index = 0

        self.manual_start_idx = ref_start
        self.manual_end_idx = ref_end
        self.manual_start_input = None
        self.manual_end_input = None
        self.preview_mode = False

        self.phase_ref_start_idx = None

        self.ref_update_cache = RefUpdateCacheManager()

        self._init_ui()
        self.pca_index = 0
        self.plot_pca_segment(self.pca_segments[0], segment_index=0)

    def _init_ui(self):
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 10
        axis = gl.GLAxisItem()
        axis.setSize(x=3, y=3, z=3)
        self.view.addItem(axis)

        self.pca_line = None
        self.ref_lines = []
        self.preview_line = None

        self.start_time_input = QLineEdit("0.0")
        initial_end_time = min(self.timestamps[-1], 10.0)
        self.end_time_input = QLineEdit(f"{initial_end_time:.3f}")
        self.segment_duration_input = QLineEdit(str(self.segment_duration))
        self.update_interval_input = QLineEdit(str(self.update_interval))

        self.min_time_label = QLabel(f"(Min: {self.timestamps[0]:.3f}s)")
        self.max_time_label = QLabel(f"(Max: {self.timestamps[-1]:.3f}s)")
        self.min_time_label.setStyleSheet("font-size: 10px; color: gray;")
        self.max_time_label.setStyleSheet("font-size: 10px; color: gray;")
        self.alpha_input = QLineEdit(str(self.alpha))
        self.fraction_input = QLineEdit(str(self.fraction))
        self.no_update_checkbox = QtWidgets.QCheckBox("No Update")

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

        self.manual_start_input = ArrowNudgeLineEdit(str(self.manual_start_idx), on_nudge=self._nudge_index)
        self.manual_end_input = ArrowNudgeLineEdit(str(self.manual_end_idx), on_nudge=self._nudge_index)
        self.manual_start_label = QLabel("Start Index:")
        self.manual_end_label = QLabel("End Index:")
        self.preview_button = QPushButton("Preview Manual Ref")
        self.preview_button.clicked.connect(self.preview_manual_ref)
        self.confirm_button = QPushButton("Confirm Manual Ref")
        self.confirm_button.clicked.connect(self.confirm_manual_ref)

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

        manual_layout = QHBoxLayout()
        for widget in [
            self.manual_start_label, self.manual_start_input,
            self.manual_end_label, self.manual_end_input,
            self.preview_button, self.confirm_button
        ]:
            manual_layout.addWidget(widget)

        fraction_alpha_layout = QHBoxLayout()
        for widget in [
            QLabel("Fraction:"), self.fraction_input,
            QLabel("Alpha:"), self.alpha_input
        ]:
            fraction_alpha_layout.addWidget(widget)

        fraction_alpha_layout.addWidget(self.no_update_checkbox)

        self.compare_button = QPushButton("Compare PCA0 vs Ref[Phase]")
        self.compare_button.clicked.connect(self.compare_pca_to_ref_phase)
        button_layout.addWidget(self.compare_button)

        self.recompute_button = QPushButton("Set PCA Window + Update Ref")
        self.recompute_button.clicked.connect(self.recompute_segments_and_ref)
        button_layout.addWidget(self.recompute_button)

        self.phase_button = QPushButton("Phase Viewer")
        self.phase_button.clicked.connect(self.launch_phase_viewer)
        button_layout.addWidget(self.phase_button)

        self.speed_button = QPushButton("Speed Viewer")
        self.speed_button.clicked.connect(self.launch_speed_viewer)
        button_layout.addWidget(self.speed_button)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(button_layout)
        layout.addLayout(fraction_alpha_layout)
        layout.addLayout(redo_layout)
        layout.addLayout(manual_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        QShortcut(QKeySequence(QtCore.Qt.Key.Key_Left), self).activated.connect(self.prev_segment)
        QShortcut(QKeySequence(QtCore.Qt.Key.Key_Right), self).activated.connect(self.next_segment)

        self.installEventFilter(self)

    def _nudge_index(self, line_edit, step):
        try:
            val = int(line_edit.text())
            new_val = val + step
            if new_val != val:
                line_edit.setText(str(new_val))
            if self.preview_mode:
                self._update_preview_line()
        except ValueError:
            pass

    def preview_manual_ref(self):
        try:
            if not self.preview_mode:
                self.preview_mode = True
                self.preview_button.setText("Cancel Preview")

                if hasattr(self, "ref_lines"):
                    for line in self.ref_lines:
                        line.setVisible(False)

                self._update_preview_line()
            else:
                self.preview_mode = False
                self.preview_button.setText("Preview Manual Ref")

                if hasattr(self, "ref_lines"):
                    for line in self.ref_lines:
                        line.setVisible(True)

                if self.preview_line and self.preview_line in self.view.items:
                    self.view.removeItem(self.preview_line)
                self.preview_line = None
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Manual Preview Error", f"Failed to toggle preview mode:\n{e}")

    def _update_preview_line(self):
        try:
            start_idx = int(self.manual_start_input.text())
            end_idx = int(self.manual_end_input.text())
            if not (0 <= start_idx < end_idx <= len(self.pca)):
                return
            ref = self.pca[start_idx:end_idx]
            if self.preview_line and self.preview_line in self.view.items:
                self.view.removeItem(self.preview_line)
            self.preview_line = gl.GLLinePlotItem(pos=ref, color=(1, 0.5, 0, 1), width=2.0, antialias=True)
            self.view.addItem(self.preview_line)
        except Exception:
            pass

    def update_ref_selector(self):
        self.ref_selector.blockSignals(True)
        self.ref_selector.clear()
        self.computed_refs = sorted(self.computed_refs, key=lambda r: r[0])
        self.computed_refs_bound = sorted(self.computed_refs_bound, key=lambda r: r[0])
        valid_set = set(ref[0] for ref in self.valid_refs) if hasattr(self, 'valid_refs') else set()

        for i, (idx, _) in enumerate(self.computed_refs):
            timestamp = self.timestamps[idx]
            label = f"Ref {i+1} @ {timestamp:.3f}s"
            if idx not in valid_set:
                label += " (not in range)"
            self.ref_selector.addItem(label)

        self.ref_selector.blockSignals(False)

    def jump_to_ref_cycle(self, index):
        if index < 0 or index >= len(self.computed_refs):
            return

        ref_start_idx, _ = self.computed_refs[index]

        # Show warning if this ref is not part of valid_refs
        if not any(ref_start_idx == r[0] for r in getattr(self, 'valid_refs', [])):
            QtWidgets.QMessageBox.warning(
                self, "Ref Not in Range", 
                f"Ref {index+1} is outside the current PCA window and won't be used in phase or segment updates."
            )

        closest_segment_index = min(
            range(len(self.pca_segments)),
            key=lambda i: abs(
                np.searchsorted(self.timestamps, (self.pca_segments[i][1] + self.pca_segments[i][2]) / 2)
                - ref_start_idx
            )
        )
        self.pca_index = closest_segment_index
        self.plot_pca_segment(self.pca_segments[self.pca_index], segment_index=self.pca_index)

    def plot_pca_segment(self, pca_segment, segment_index=None):
        try:
            X_pca, seg_start_time, seg_end_time = pca_segment

            if segment_index is not None:
                self.pca_index = segment_index

            if self.pca_line and self.pca_line in self.view.items:
                self.view.removeItem(self.pca_line)

            if hasattr(self, "ref_lines"):
                for line in self.ref_lines:
                    if line in self.view.items:
                        self.view.removeItem(line)
            else:
                self.ref_lines = []

            self.ref_lines = []

            # Plot PCA segment
            self.pca_line = gl.GLLinePlotItem(pos=X_pca, color=(0, 0, 1, 1), width=2.0, antialias=True)
            self.view.addItem(self.pca_line)

            # Plot relevant refs
            refs_in_segment = [
                (idx, cycle)
                for (idx, cycle) in self.all_valid_refs
                if seg_start_time <= self.timestamps[idx] <= seg_end_time
            ]
            if not refs_in_segment and self.all_valid_refs:
                closest_ref = min(
                    self.all_valid_refs,
                    key=lambda rc: abs(self.timestamps[rc[0]] - seg_start_time)
                )
                refs_in_segment = [closest_ref]

            ref_colors = [(1, 0, 0, 1), (0, 0.6, 0, 1), (1, 0.5, 0, 1), (0.5, 0, 1, 1)]
            for i, (_, ref_cycle) in enumerate(refs_in_segment):
                color = ref_colors[i % len(ref_colors)]
                line = gl.GLLinePlotItem(pos=ref_cycle, color=color, width=2.0, antialias=True)
                self.view.addItem(line)
                self.ref_lines.append(line)

            # Compute index range for the segment
            seg_start_idx = np.searchsorted(self.timestamps, seg_start_time, side='left')
            seg_end_idx = np.searchsorted(self.timestamps, seg_end_time, side='right')

            # Try to find computed ref bound inside segment
            matched_bound = None
            for bound in self.computed_refs_bound:
                if seg_start_idx <= bound[0] and bound[1] <= seg_end_idx:
                    matched_bound = bound
                    break

            if matched_bound:
                self.manual_start_input.setText(str(matched_bound[0]))
                self.manual_end_input.setText(str(matched_bound[1]))
            else:
                self.manual_start_input.setText(str(seg_start_idx))
                self.manual_end_input.setText(str(seg_start_idx + 1000))

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

            self.phase0 = np.array([], dtype=np.uint8)
            self.updated_refs = []

            start_time = float(self.start_time_input.text())
            end_time = float(self.end_time_input.text())
            self.segment_duration = float(self.segment_duration_input.text())
            self.update_interval = float(self.update_interval_input.text())
            self.alpha = float(self.alpha_input.text())
            self.fraction = float(self.fraction_input.text())
            no_update = self.no_update_checkbox.isChecked()

            self.segment_size = int(self.segment_duration * SAMPLING_RATE)

            if start_time >= end_time:
                raise ValueError("Start time must be less than end time")

            pca_start_idx = np.searchsorted(self.timestamps, start_time, side='left')
            pca_end_idx = np.searchsorted(self.timestamps, end_time, side='right')

            valid_refs = []
            ref_index_list = []

            # Build a mapping from start index to the last (index, bound) pair
            start_idx_to_last = {}

            for i in range(len(self.computed_refs_bound)):
                start, end = self.computed_refs_bound[i]
                if start >= pca_start_idx and end <= pca_end_idx:
                    start_idx_to_last[start] = (i, self.computed_refs_bound[i])

            # Now collect only the final ones for each unique start
            for i, bound in start_idx_to_last.values():
                valid_refs.append(self.computed_refs[i])
                ref_index_list.append(bound)
                
            self.valid_refs = valid_refs

            if not self.valid_refs:
                QtWidgets.QMessageBox.warning(self, "No Valid Ref", "No valid reference cycles found in the selected time range.")
                return

            for i in range(len(ref_index_list) - 1):
                pair = (ref_index_list[i], ref_index_list[i + 1])
                cached = self.ref_update_cache.get_entry(pair, self.update_interval, self.alpha, self.fraction, no_update=no_update)

                if cached:
                    self.updated_refs += cached.updated_refs
                    self.phase0 = np.concatenate((self.phase0, cached.phase0))
                    continue
                else:
                    windowed_pca_cache = self.pca[self.valid_refs[i][0]:self.valid_refs[i+1][0]]
                    updated_refs, phase0 = ref_cycle_update(
                        windowed_pca_cache, self.valid_refs[i],
                        update_interval=self.update_interval,
                        fraction=self.fraction,
                        alpha=self.alpha,
                        no_update=no_update,
                    )
                    self.updated_refs += updated_refs
                    self.phase0 = np.concatenate((self.phase0, phase0))
                    entry = SegmentCacheEntry(
                        updated_refs=updated_refs,
                        phase0=phase0,
                    )
                    self.ref_update_cache.add_entry(
                        pair, self.update_interval, self.alpha, self.fraction, entry, no_update=no_update
                    )

            windowed_pca_cache = self.pca[self.valid_refs[-1][0]:pca_end_idx]
            updated_refs, phase0 = ref_cycle_update(
                windowed_pca_cache, self.valid_refs[-1],
                update_interval=self.update_interval,
                fraction=self.fraction,
                alpha=self.alpha,
                no_update=no_update,
            )
            self.updated_refs += updated_refs
            self.phase0 = np.concatenate((self.phase0, phase0))

            self.phase_time = self.timestamps[self.valid_refs[0][0]:pca_end_idx]
            phase = self.phase0 / REFERENCE_NUM_POINTS * 2 * np.pi
            raw_phase = np.unwrap(phase)
            self.phase = savgol_filter(raw_phase, window_length=51, polyorder=3)
            self.phase_ref_start_idx = self.valid_refs[0][0]
            self.all_valid_refs = sorted(self.valid_refs + self.updated_refs, key=lambda r: r[0])
            self.update_ref_selector()

            save_pca_metadata(
                self.file_basename,
                self.computed_refs_bound,
                self.phase_time,
                self.phase0
            )

            segments = []
            windowed_pca = self.pca[self.valid_refs[0][0]:pca_end_idx]
            for i in range(0, len(windowed_pca), self.segment_size):
                seg = windowed_pca[i:i+self.segment_size]
                t0 = self.timestamps[pca_start_idx + i]
                t1 = self.timestamps[pca_start_idx + i + len(seg) - 1]
                segments.append((seg, t0, t1))

            self.pca_segments = segments
            self.pca_index = 0
            if self.pca_segments:
                self.plot_pca_segment(self.pca_segments[0], segment_index=0)

            # Debugging information
            # print(f"[DEBUG] phase_time len: {len(self.phase_time)}")
            # print(f"[DEBUG] phase0 len: {len(self.phase0)}")
            # print(f"[DEBUG] mismatch = {len(self.phase_time) != len(self.phase0)}")
            # print(f"[DEBUG] updated ref starts: {[r[0] for r in self.updated_refs]}")
            # print(f"[DEBUG] valid_ref starts: {[r[0] for r in self.valid_refs]}")
            # print(f"[DEBUG] ref_index_list: {ref_index_list}")
            # print(f"[DEBUG] total timestamp range: {pca_end_idx - self.valid_refs[0][0]}")

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
            self.computed_refs_bound[ref_index] = (new_ref_start, new_ref_end)
            self.recompute_segments_and_ref()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ref Redo Error", f"Failed to redo ref cycle:\n{e}")

    def remove_selected_ref(self):
        ref_index = self.ref_selector.currentIndex()
        if 0 <= ref_index < len(self.computed_refs):
            self.computed_refs.pop(ref_index)
            self.computed_refs_bound.pop(ref_index)
            self.updated_refs = []

            self.update_ref_selector()

            if not self.computed_refs:
                # Clear all reference lines from the view
                for line in self.ref_lines:
                    if line in self.view.items:
                        self.view.removeItem(line)
                self.ref_lines = []
                self.valid_refs = []
                self.all_valid_refs = []

                # # Remove preview if showing
                # if self.preview_line and self.preview_line in self.view.items:
                #     self.view.removeItem(self.preview_line)
                #     self.preview_line = None
                #     self.preview_mode = False
                #     self.preview_button.setText("Preview Manual Ref")

                # Keep showing current PCA segment (just without ref)
                if self.pca_segments:
                    self.plot_pca_segment(self.pca_segments[self.pca_index])
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
            self.computed_refs_bound.append((new_ref_start, new_ref_end))
            self.recompute_segments_and_ref()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Add Ref Error", f"Failed to add new ref cycle:\n{e}")

    def confirm_manual_ref(self):
        try:
            start_idx = int(self.manual_start_input.text())
            end_idx = int(self.manual_end_input.text())
            if start_idx < 0 or end_idx > len(self.pca) or start_idx >= end_idx:
                return
            ref = self.pca[start_idx:end_idx]
            smooth_ref = smooth_trajectory(ref)
            self.computed_refs.append((start_idx, smooth_ref))
            self.computed_refs_bound.append((start_idx, end_idx))
            if self.preview_line and self.preview_line in self.view.items:
                self.view.removeItem(self.preview_line)
            self.preview_line = None
            self.preview_mode = False
            self.preview_button.setText("Preview Manual Ref")
            self.recompute_segments_and_ref()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Manual Confirm Error", f"Failed to confirm manual ref cycle:\n{e}")

    def prev_segment(self):
        if self.pca_index > 0:
            self.pca_index -= 1
            self.plot_pca_segment(self.pca_segments[self.pca_index], segment_index=self.pca_index)


    def next_segment(self):
        if self.pca_index < len(self.pca_segments) - 1:
            self.pca_index += 1
            self.plot_pca_segment(self.pca_segments[self.pca_index], segment_index=self.pca_index)

    def compare_pca_to_ref_phase(self):
        try:
            if not hasattr(self, 'phase') or not hasattr(self, 'phase_time') or self.phase_ref_start_idx is None:
                QtWidgets.QMessageBox.warning(self, "Missing Phase", "No phase data available. Please recompute first.")
                return

            times = self.phase_time
            phase = self.phase0
            if len(phase) != len(times):
                QtWidgets.QMessageBox.critical(self, "Mismatch", "Phase and time arrays have different lengths.")
                return

            if not self.all_valid_refs:
                QtWidgets.QMessageBox.warning(self, "No Ref", "No reference cycles to use.")
                return

            ref_trace = np.zeros_like(phase, dtype=float)
            indices = np.arange(len(phase))
            start_indices = [start - self.phase_ref_start_idx for start, _ in self.all_valid_refs] + [len(phase)]

            for i in range(len(self.all_valid_refs)):
                start_idx = start_indices[i]
                end_idx = start_indices[i + 1]
                mask = (indices >= start_idx) & (indices < end_idx)

                ref_cycle = self.all_valid_refs[i][1]

                if not np.any(mask):
                    continue

                if np.max(phase[mask]) >= len(ref_cycle):
                    QtWidgets.QMessageBox.warning(self, "Ref too short", f"Reference cycle {i+1} is too short for assigned phases.")
                    return

                ref_trace[mask] = ref_cycle[phase[mask], 0]

            pca_trace = self.pca[self.phase_ref_start_idx:self.phase_ref_start_idx + len(phase), 0]

            self.comparison_viewer = PCAComparisonViewer(times, pca_trace, ref_trace)
            self.comparison_viewer.show()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Compare Error", f"Comparison failed:{e}")

    def launch_phase_viewer(self):
        if not hasattr(self, "phase") or not hasattr(self, "phase_time"):
            QtWidgets.QMessageBox.warning(self, "Missing Phase", "Phase data not available. Run recompute first.")
            return
        self.phase_viewer = PhaseViewer(self.phase_time, self.phase)
        self.phase_viewer.show()

    def launch_speed_viewer(self):
        if not hasattr(self, "phase") or not hasattr(self, "phase_time"):
            QtWidgets.QMessageBox.warning(self, "Missing Phase", "Phase data not available. Run recompute first.")
            return

        self.speed_viewer = PCASpeedViewer(
            phase=self.phase,
            phase_time=self.phase_time,
            sampling_rate=1 / SAMPLING_RATE
        )
        self.speed_viewer.show()

class ArrowNudgeLineEdit(QLineEdit):
    def __init__(self, *args, on_nudge=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_nudge = on_nudge  # Callback function with direction

    def keyPressEvent(self, event):
        key = event.key()
        step = 50 if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier else 1
        if key == QtCore.Qt.Key.Key_Up:
            if self.on_nudge:
                self.on_nudge(self, step)
            event.accept()
        elif key == QtCore.Qt.Key.Key_Down:
            if self.on_nudge:
                self.on_nudge(self, -step)
            event.accept()
        else:
            super().keyPressEvent(event)

def save_pca_metadata(file_basename, computed_ref_bound, phase_time, phase0):
    """
    Save minimal PCA metadata to disk for future speed analysis.
    Only saves:
    - computed_ref_bound (to .json)
    - phase_time and phase0 (to compressed .npz)
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    save_dir = os.path.join(project_root, "results", file_basename)
    os.makedirs(save_dir, exist_ok=True)

    # --- Diagnostic print for data types and size ---
    print("[INFO] Saving phase metadata:")
    print(f"  - phase_time dtype: {phase_time.dtype}, shape: {phase_time.shape}, size: {phase_time.nbytes / 1e6:.2f} MB")
    print(f"  - phase0 dtype: {phase0.dtype}, shape: {phase0.shape}, size: {phase0.nbytes / 1e6:.2f} MB")

    # Save only what's needed for downstream speed analysis
    np.savez_compressed(
        os.path.join(save_dir, "phase_data.npz"),
        phase_time=phase_time,
        phase0=phase0,
    )

    # Save ref bounds as JSON
    ref_bounds_serializable = [(int(start), int(end)) for start, end in computed_ref_bound]
    with open(os.path.join(save_dir, "ref_bounds.json"), "w") as f:
        json.dump(ref_bounds_serializable, f, indent=2)

    print(f"[SAVE] Metadata saved to: {save_dir}")