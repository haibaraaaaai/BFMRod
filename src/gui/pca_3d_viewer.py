from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
)
import numpy as np
from processing.pca_module import run_pca_workflow
from config import SAMPLING_RATE, DEFAULT_PCA_SEGMENT_DURATION, DEFAULT_CLOSURE_THRESHOLD

class PCA3DViewer(QMainWindow):
    def __init__(self, data, timestamps):
        super().__init__()
        self.setWindowTitle("PCA 3D Viewer")
        self.setGeometry(200, 200, 800, 600)

        self.data = data
        self.timestamps = timestamps
        self.segment_duration = DEFAULT_PCA_SEGMENT_DURATION
        self.closure_threshold = DEFAULT_CLOSURE_THRESHOLD

        # Run PCA on full data
        self.pca_results = run_pca_workflow(
            data=self.data,
            timestamps=self.timestamps,
            start_time=self.timestamps[0],
            end_time=self.timestamps[-1],
            segment_duration=self.segment_duration,
            closure_threshold=self.closure_threshold
        )

        # Placeholder UI: display data info and buttons
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # Info label
        pca_segments = self.pca_results["pca_segments"]
        n_segments = len(pca_segments) if pca_segments else 0
        info = (
            f"Total samples: {len(self.data)}\n"
            f"PCA segments generated: {n_segments}\n"
            f"Showing initial 1s PCA segment..."
        )
        self.info_label = QLabel(info)
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # Placeholder buttons for next features
        button_layout = QHBoxLayout()

        self.detect_button = QPushButton("Re-detect Ref Cycle")
        self.add_manual_button = QPushButton("Add Manual Ref")
        self.remove_button = QPushButton("Remove Manual Ref")
        self.recompute_button = QPushButton("Recompute Phase")
        self.launch_phase_button = QPushButton("Phase Viewer")
        self.launch_speed_button = QPushButton("Speed Viewer")

        for btn in [
            self.detect_button,
            self.add_manual_button,
            self.remove_button,
            self.recompute_button,
            self.launch_phase_button,
            self.launch_speed_button
        ]:
            button_layout.addWidget(btn)

        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
