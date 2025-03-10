from PyQt6.QtWidgets import QMainWindow, QPushButton, QFileDialog, QApplication
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("BFM Analysis GUI")
        self.setGeometry(100, 100, 400, 300)

        self.open_button = QPushButton("Open TDMS", self)
        self.open_button.setGeometry(150, 130, 100, 40)
        self.open_button.clicked.connect(self.open_tdms_file)

    def open_tdms_file(self):
        """Opens a file dialog to select a TDMS file and loads it into a new window."""
        from gui.tdms_viewer import TDMSViewer

        file_path, _ = QFileDialog.getOpenFileName(self, "Select TDMS File", "", "TDMS Files (*.tdms)")

        if file_path:
            self.tdms_viewer = TDMSViewer(file_path)
            self.tdms_viewer.show()
            self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
