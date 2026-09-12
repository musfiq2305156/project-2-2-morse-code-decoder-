"""
Phase 6 smoke test: same headless decode-drive as Phase 5's test, but
additionally grabs a screenshot of the rendered window afterward so the
waveform/envelope/threshold plot can be visually inspected, not just
checked for "didn't crash".

Usage (headless, no display attached):
    QT_QPA_PLATFORM=offscreen python scripts/test_phase6_waveform.py test_signal.wav out.png
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import STYLESHEET


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/test_phase6_waveform.py <path_to_wav> <screenshot_out.png>")
        sys.exit(1)

    wav_path = sys.argv[1]
    out_path = sys.argv[2]

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()

    window._start_decode(wav_path)
    window._worker.wait()
    app.processEvents()

    # A couple more event-loop pumps so matplotlib's draw_idle() actually
    # renders before we grab the pixels.
    for _ in range(5):
        app.processEvents()

    pixmap = window.grab()
    pixmap.save(out_path)

    print(f"status label   : {window.status_label.text()}")
    print(f"decoded label  : {window.decoded_text_label.text()}")
    print(f"screenshot     : {out_path}")


if __name__ == "__main__":
    main()
