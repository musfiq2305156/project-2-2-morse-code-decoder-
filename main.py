"""
Entry point. Run with:  python main.py
"""

import sys
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import STYLESHEET


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
