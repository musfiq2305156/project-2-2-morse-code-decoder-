"""
Phase 4 smoke test: the complete headless core pipeline.

    .wav file -> audio_io -> signal_processing -> morse_state_machine
              -> morse_decoder -> decoded text

No GUI involved anywhere in this chain - this is the same pipeline the
UI will call starting in Phase 5, just driven from the command line.

Usage:
    python scripts/make_test_wav.py test_signal.wav
    python scripts/test_phase4_decoder.py test_signal.wav
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.audio_io import load_wav_file
from core.signal_processing import analyze
from core.morse_state_machine import signal_to_morse
from core.morse_decoder import decode_morse


def main():
    parser = argparse.ArgumentParser(description="Phase 4: full WAV-to-text pipeline test")
    parser.add_argument("wav_path", help="Path to a .wav file")
    parser.add_argument("--window", type=float, default=10.0, help="Envelope moving-average window in ms")
    parser.add_argument("--ratio", type=float, default=0.3, help="Threshold as fraction of peak envelope")
    args = parser.parse_args()

    buffer = load_wav_file(args.wav_path)
    phase2 = analyze(buffer, window_ms=args.window, threshold_ratio=args.ratio)
    phase3 = signal_to_morse(phase2["binary"], buffer.sample_rate)
    text = decode_morse(phase3["morse"])

    print(f"file             : {args.wav_path}")
    print(f"estimated unit   : {phase3['unit_seconds'] * 1000:.1f} ms")
    print(f"morse notation   : {phase3['morse']}")
    print(f"decoded text     : {text}")


if __name__ == "__main__":
    main()
