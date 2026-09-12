"""
Validates core.live_decoder.IncrementalMorseDecoder against the known-good
batch pipeline (core.pipeline.decode_wav_file) by feeding the exact same
audio through both: once as a single call (batch), and once split into
small chunks pushed one at a time (streaming, exactly as ui.live_worker
will do it against a real/simulated microphone).

Run: python scripts/test_live_decoder.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.io import wavfile

from core.pipeline import decode_wav_file
from core.live_decoder import IncrementalMorseDecoder
from core.morse_decoder import encode_text

SAMPLE_RATE = 44100
TONE_HZ = 700


def make_wav(text: str, wpm: float = 18, noise_amplitude: float = 0.0,
             path: str = "/tmp/live_decoder_test.wav") -> str:
    unit = 1.2 / wpm
    morse = encode_text(text)
    rng = np.random.default_rng(0)
    chunks = []

    def tone(duration):
        n = int(duration * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE
        return 0.5 * np.sin(2 * np.pi * TONE_HZ * t)

    def silence(duration):
        n = int(duration * SAMPLE_RATE)
        return np.zeros(n)

    chunks.append(silence(unit * 4))  # leading silence, like a real recording
    for word_i, word in enumerate(morse.split(" / ")):
        if word_i > 0:
            chunks.append(silence(unit * 7))
        for letter_i, letter in enumerate(word.split(" ")):
            if letter_i > 0:
                chunks.append(silence(unit * 3))
            for sym_i, sym in enumerate(letter):
                if sym_i > 0:
                    chunks.append(silence(unit))
                chunks.append(tone(unit if sym == "." else unit * 3))
    chunks.append(silence(unit * 4))

    signal = np.concatenate(chunks)
    if noise_amplitude > 0:
        signal = signal + rng.normal(0, noise_amplitude, size=signal.shape)
    signal = np.clip(signal, -1.0, 1.0)
    int16 = (signal * 32767).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, int16)
    return path


def run_case(label: str, text: str, wpm: float, noise_amplitude: float,
             chunk_size: int) -> bool:
    path = make_wav(text, wpm=wpm, noise_amplitude=noise_amplitude)
    batch = decode_wav_file(path, tone_freq_hz=TONE_HZ)

    _, samples_int16 = wavfile.read(path)
    samples = samples_int16.astype(np.float64) / 32767.0

    decoder = IncrementalMorseDecoder(SAMPLE_RATE, tone_freq_hz=TONE_HZ)
    for start in range(0, len(samples), chunk_size):
        decoder.push(samples[start:start + chunk_size])
    decoder.flush()

    ok = decoder.text.strip() == text.strip()
    print(f"[{label}] expected={text!r} batch={batch['text']!r} "
          f"streaming={decoder.text!r} chunk={chunk_size} noise={noise_amplitude} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    # Clean signal, various chunk sizes (mimics different poll intervals).
    for chunk_size in (256, 1024, 4096, 17640):  # 17640 = 0.4s at 44.1kHz
        results.append(run_case("clean", "TIGER HELP", wpm=18,
                                 noise_amplitude=0.0, chunk_size=chunk_size))
    # With realistic mic-style background noise.
    for noise in (0.02, 0.05):
        results.append(run_case("noisy", "SOS TEST", wpm=20,
                                 noise_amplitude=noise, chunk_size=1024))

    print()
    if all(results):
        print(f"All {len(results)} cases passed.")
    else:
        print(f"{results.count(False)} of {len(results)} cases FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
