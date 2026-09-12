"""
Generates a simple test .wav file: a 700 Hz tone pulsing on/off, roughly
approximating a Morse-like on/off pattern. This is only a synthetic
smoke-test signal for Phase 1 (audio loading) - it is NOT yet decoded
by a real Morse pipeline; that logic arrives in Phases 2-4.

Usage:
    python scripts/make_test_wav.py [output_path]
"""

import sys
import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 44100
TONE_HZ = 700


def make_pulse_pattern(pattern_seconds, sample_rate=SAMPLE_RATE, tone_hz=TONE_HZ):
    """
    pattern_seconds: list of (duration_seconds, is_tone_on) tuples.
    Returns int16 numpy array.
    """
    chunks = []
    for duration, is_on in pattern_seconds:
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        if is_on:
            chunk = 0.6 * np.sin(2 * np.pi * tone_hz * t)
        else:
            chunk = np.zeros(n_samples)
        chunks.append(chunk)
    signal = np.concatenate(chunks) if chunks else np.zeros(0)
    return (signal * np.iinfo(np.int16).max).astype(np.int16)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "test_signal.wav"

    # dot=0.1s, dash=0.3s, intra-gap=0.1s, char-gap=0.3s -> roughly "S O S"
    unit = 0.1
    pattern = [
        (unit, True), (unit, False), (unit, True), (unit, False), (unit, True),  # S
        (unit * 3, False),
        (unit * 3, True), (unit, False), (unit * 3, True), (unit, False), (unit * 3, True),  # O
        (unit * 3, False),
        (unit, True), (unit, False), (unit, True), (unit, False), (unit, True),  # S
    ]

    samples = make_pulse_pattern(pattern)
    wavfile.write(out_path, SAMPLE_RATE, samples)
    print(f"Wrote {len(samples)} samples ({len(samples) / SAMPLE_RATE:.2f}s) to '{out_path}'")


if __name__ == "__main__":
    main()
