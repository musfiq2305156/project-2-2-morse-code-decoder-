"""
Phase 5 smoke test: exercises the actual UI decode flow (not just the
core pipeline) headlessly - it instantiates the real MainWindow, calls
the same method the "Load WAV File" button triggers, waits for the
background DecodeWorker thread to finish, and asserts the on-screen
labels updated with the correct decoded text.

This is the closest thing to an automated UI test without a real
display. Run with QT_QPA_PLATFORM=offscreen (already set in the
command below) since this environment has no monitor attached.

Usage:
    QT_QPA_PLATFORM=offscreen python scripts/test_phase5_ui.py test_signal.wav SOS
    QT_QPA_PLATFORM=offscreen python scripts/test_phase5_ui.py hi_ok.wav "HI OK"
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_phase5_ui.py <path_to_wav> [expected_text]")
        sys.exit(1)

    wav_path = sys.argv[1]
    expected_text = sys.argv[2] if len(sys.argv) > 2 else None

    app = QApplication(sys.argv)
    window = MainWindow()

    # Simulate what clicking "Load WAV File" + picking a file does,
    # skipping the actual QFileDialog since there's no display to
    # click through in this headless test.
    window._start_decode(wav_path)
    window._worker.wait()  # block until the background thread's run() returns
    app.processEvents()    # pump the event loop so the queued finished/failed
                            # signal is actually delivered to the main-thread slot

    decoded = window.decoded_text_label.text()
    morse = window.morse_label.text()
    status = window.status_label.text()

    print(f"status label   : {status}")
    print(f"decoded label  : {decoded}")
    print(f"morse label    : {morse}")
    print(f"button enabled : {window.load_button.isEnabled()}")

    if expected_text is not None:
        assert decoded == expected_text, f"Expected '{expected_text}', got '{decoded}'"
        print(f"PASS: decoded text matches expected '{expected_text}'")

    assert status == "Done", f"Expected status 'Done', got '{status}'"
    assert window.load_button.isEnabled(), "Load button should be re-enabled after decoding"
    print("PASS: UI state correct after decode")


if __name__ == "__main__":
    main()
