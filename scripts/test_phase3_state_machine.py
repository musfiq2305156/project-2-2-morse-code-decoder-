"""
Phase 3 smoke test: loads a WAV, runs it through Phase 2 (envelope +
thresholding) then Phase 3 (run-length encoding + timing classification),
and prints the resulting Morse notation string plus the per-run timing
breakdown so misclassifications are easy to debug.

Usage:
    python scripts/make_test_wav.py test_signal.wav
    python scripts/test_phase3_state_machine.py test_signal.wav
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.audio_io import load_wav_file
from core.signal_processing import analyze
from core.morse_state_machine import signal_to_morse


def main():
    parser = argparse.ArgumentParser(description="Phase 3: timing classification test")
    parser.add_argument("wav_path", help="Path to a .wav file")
    parser.add_argument("--window", type=float, default=10.0, help="Envelope moving-average window in ms")
    parser.add_argument("--ratio", type=float, default=0.3, help="Threshold as fraction of peak envelope")
    parser.add_argument("--verbose", action="store_true", help="Print each classified run")
    args = parser.parse_args()

    buffer = load_wav_file(args.wav_path)
    phase2 = analyze(buffer, window_ms=args.window, threshold_ratio=args.ratio)
    phase3 = signal_to_morse(phase2["binary"], buffer.sample_rate)

    print(f"estimated unit   : {phase3['unit_seconds'] * 1000:.1f} ms")
    print(f"morse notation   : {phase3['morse']}")

    if args.verbose:
        print("\nper-run breakdown:")
        unit = phase3["unit_seconds"]
        for run in phase3["runs"]:
            kind = "TONE " if run.is_tone else "SILENCE"
            print(f"  {kind}  {run.duration_seconds * 1000:7.1f} ms  "
                  f"({run.duration_seconds / unit:.2f} units)")


if __name__ == "__main__":
    main()
