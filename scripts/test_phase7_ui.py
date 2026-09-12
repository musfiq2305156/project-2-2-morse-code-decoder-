"""
Phase 7 smoke test: drives the "Simulate from File" live flow through the
actual MainWindow (bypassing the QFileDialog, same approach as Phase 5's
test), runs the Qt event loop for a few seconds of simulated live audio,
checks the decoded text and button states mid-session, then stops the
session and checks everything resets correctly.

Usage:
    QT_QPA_PLATFORM=offscreen python scripts/test_phase7_ui.py test_signal.wav SOS screenshot.png
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QEventLoop, QTimer

from ui.main_window import MainWindow
from ui.theme import STYLESHEET
from core.audio_io import SimulatedMicrophoneStream


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/test_phase7_ui.py <wav_path> <expected_text> [screenshot_out.png]")
        sys.exit(1)

    wav_path = sys.argv[1]
    expected_text = sys.argv[2]
    screenshot_path = sys.argv[3] if len(sys.argv) > 3 else None

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()

    # Bypass the QFileDialog (no display to click through) the same way
    # window._on_simulate_clicked() would after the user picks a file.
    window._start_live(SimulatedMicrophoneStream(wav_path), listening_label="Listening\u2026 (test)")

    print(f"mid-session mic button       : '{window.mic_button.text()}' enabled={window.mic_button.isEnabled()}")
    print(f"mid-session simulate button  : '{window.simulate_button.text()}' enabled={window.simulate_button.isEnabled()}")
    print(f"mid-session load button      : enabled={window.load_button.isEnabled()}")

    assert not window.load_button.isEnabled(), "Load button should be disabled during a live session"
    assert not window.mic_button.isEnabled(), "Mic button should be disabled while simulate is running"
    assert window.simulate_button.text() == "Stop Simulation"
    print("PASS: controls correctly locked during live session")

    loop = QEventLoop()
    QTimer.singleShot(4000, loop.quit)
    loop.exec()

    if screenshot_path:
        for _ in range(5):
            app.processEvents()
        window.grab().save(screenshot_path)
        print(f"screenshot saved: {screenshot_path}")

    decoded = window.decoded_text_label.text()
    print(f"decoded text after 4s of simulated live audio: '{decoded}'")
    assert decoded == expected_text, f"Expected '{expected_text}', got '{decoded}'"
    print(f"PASS: live decode converged to expected '{expected_text}'")

    window._stop_live()
    app.processEvents()

    print(f"post-stop mic button        : '{window.mic_button.text()}' enabled={window.mic_button.isEnabled()}")
    print(f"post-stop simulate button   : '{window.simulate_button.text()}' enabled={window.simulate_button.isEnabled()}")
    print(f"post-stop load button       : enabled={window.load_button.isEnabled()}")
    print(f"post-stop status            : '{window.status_label.text()}'")

    assert window.load_button.isEnabled()
    assert window.mic_button.isEnabled()
    assert window.simulate_button.isEnabled()
    assert window.simulate_button.text() == "Simulate from File"
    assert window._live_worker is None
    print("PASS: controls correctly reset after stopping live session")


if __name__ == "__main__":
    main()
