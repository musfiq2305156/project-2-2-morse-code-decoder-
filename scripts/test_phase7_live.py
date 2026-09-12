"""
Phase 7 smoke test: exercises LiveDecodeWorker end-to-end using
SimulatedMicrophoneStream (replaying a .wav file as if it were a live
mic), since this environment has no physical microphone. Verifies the
rolling buffer accumulates correctly and the decoded text/morse
converges to the correct final message as more audio arrives.

Usage:
    python scripts/make_test_wav.py test_signal.wav
    QT_QPA_PLATFORM=offscreen python scripts/test_phase7_live.py test_signal.wav SOS
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QEventLoop, QTimer

from core.audio_io import SimulatedMicrophoneStream
from ui.live_worker import LiveDecodeWorker


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_phase7_live.py <path_to_wav> [expected_text]")
        sys.exit(1)

    wav_path = sys.argv[1]
    expected_text = sys.argv[2] if len(sys.argv) > 2 else None

    app = QApplication(sys.argv)

    sim = SimulatedMicrophoneStream(wav_path, blocksize=1024)
    worker = LiveDecodeWorker(sim, poll_interval=0.3)

    updates = []
    errors = []
    worker.updated.connect(lambda result: updates.append(result))
    worker.failed.connect(lambda msg: errors.append(msg))

    worker.start()

    # Run the Qt event loop for a fixed duration so queued signals
    # actually get delivered as they arrive, then stop.
    loop = QEventLoop()
    QTimer.singleShot(4000, loop.quit)  # run for 4 seconds of simulated "live" audio
    loop.exec()

    worker.stop()

    if errors:
        print(f"FAILED signal(s) received: {errors}")
        sys.exit(1)

    print(f"received {len(updates)} live update(s)")
    for i, u in enumerate(updates):
        print(f"  update {i}: morse='{u['morse']}'  text='{u['text']}'")

    if not updates:
        print("FAIL: no updates received at all")
        sys.exit(1)

    final_text = updates[-1]["text"]
    print(f"\nfinal decoded text: '{final_text}'")

    if expected_text is not None:
        assert final_text == expected_text, f"Expected '{expected_text}', got '{final_text}'"
        print(f"PASS: final live decode matches expected '{expected_text}'")


if __name__ == "__main__":
    main()
