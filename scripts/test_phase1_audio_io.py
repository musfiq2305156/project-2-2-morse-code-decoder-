"""
Phase 1 smoke test: confirms audio_io.load_wav_file() works correctly,
independent of any GUI. Run after generating a test file:

    python scripts/make_test_wav.py test_signal.wav
    python scripts/test_phase1_audio_io.py test_signal.wav
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.audio_io import load_wav_file, describe


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_phase1_audio_io.py <path_to_wav>")
        sys.exit(1)

    path = sys.argv[1]
    buffer = load_wav_file(path)
    print(describe(buffer))


if __name__ == "__main__":
    main()
