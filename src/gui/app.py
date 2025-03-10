from PyQt6.QtWidgets import QApplication
import sys
from gui import MainWindow

def run_gui():
    """Launches the main GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
