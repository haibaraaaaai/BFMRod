"""
main.py – Entry point to launch the GUI application.

Currently launches MainWindow, which gives access to the TDMS viewer and related tools.
"""

import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
