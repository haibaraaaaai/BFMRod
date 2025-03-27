"""
main.py – Entry point to launch the GUI application.

Currently launches TDMSViewer, which gives access to the TDMS raw data and PCA entries.
"""

import sys
from PyQt6.QtWidgets import QApplication
from gui.tdms_viewer import TDMSViewer

def main():
    app = QApplication(sys.argv)
    viewer = TDMSViewer()
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
