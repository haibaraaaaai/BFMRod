"""
main_window.py – Initial window with file selection for TDMS viewer launch.
"""

import os
import sys
from PyQt6.QtWidgets import QMainWindow, QPushButton, QFileDialog, QApplication
from gui.tdms_viewer import TDMSViewer

LAST_FILE_PATH = "last_file.txt"

class MainWindow(QMainWindow):
    """
    Launch screen for the BFM Analysis GUI.

    Features:
        - Single button to open a TDMS file.
        - Launches full TDMS viewer upon file selection.
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("BFM Analysis GUI")
        self.setGeometry(100, 100, 400, 300)

        # --- Open File Button ---
        self.open_button = QPushButton("Open TDMS", self)
        self.open_button.setGeometry(150, 130, 100, 40)
        self.open_button.clicked.connect(self.open_tdms_file)

    def get_last_folder(self):
        """Return the folder of the last opened file, if available."""
        if os.path.exists(LAST_FILE_PATH):
            with open(LAST_FILE_PATH, "r") as f:
                last_path = f.read().strip()
                if os.path.exists(last_path):
                    return os.path.dirname(last_path)
        return ""

    def open_tdms_file(self):
        """Open a TDMS file and launch the viewer."""
        initial_dir = self.get_last_folder()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select TDMS File", initial_dir, "TDMS Files (*.tdms)")
        if file_path:
            with open(LAST_FILE_PATH, "w") as f:
                f.write(file_path)

            self.tdms_viewer = TDMSViewer(file_path)
            self.tdms_viewer.show()
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
