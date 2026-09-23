"""
Text -> Morse audio synthesis: the inverse of the decode pipeline.

Given text, this produces an actual AudioBuffer of a Morse tone with
standard timing (dot = 1 unit, dash = 3 units, intra-character gap = 1
unit, inter-character gap = 3 units, word gap = 7 units), so it can be
played, exported to a .wav file, or fed straight back into the app's own
decoder as a correctness check.

This is a new, standalone module - it only *reads* from morse_decoder.py
(encode_text) and config.py (wpm_to_unit_seconds), and does not modify
either. Nothing in the existing decode pipeline (audio_io, signal_processing,
morse_state_machine, pipeline) is touched or imported for writing here.
"""

from __future__ import annotations

import numpy as np
from scipy.io import wavfile

from core.config import AudioBuffer, wpm_to_unit_seconds, DEFAULT_WPM, DEFAULT_TONE_FREQ_HZ
from core.morse_decoder import encode_text

DEFAULT_AMPLITUDE = 0.5
DEFAULT_FADE_MS = 4.0  # short rise/fall time on each tone burst, see _tone_burst()


def _tone_burst(duration_seconds: float, sample_rate: int, tone_freq_hz: float,
                 amplitude: float, fade_ms: float) -> np.ndarray:
    """
    One "on" pulse: a sine wave at tone_freq_hz, with a short linear
    fade-in/fade-out.

    The fade matters for the same reason real CW (Morse) transmitters use
    a deliberate rise/fall time on their keying: an instant on/off jump
    is a discontinuity, which spreads energy across a wide range of
    frequencies (an audible "click", and - on a real radio transmitter -
    unwanted out-of-band interference called "key clicks"). A few
    milliseconds of ramp keeps the tone's energy concentrated near its
    intended frequency instead.
    """
    n = max(int(round(duration_seconds * sample_rate)), 1)
    t = np.arange(n) / sample_rate
    wave = amplitude * np.sin(2 * np.pi * tone_freq_hz * t)

    fade_n = min(int(round(fade_ms / 1000.0 * sample_rate)), n // 2)
    if fade_n > 0:
        ramp = np.linspace(0.0, 1.0, fade_n)
        wave[:fade_n] *= ramp
        wave[-fade_n:] *= ramp[::-1]

    return wave


def text_to_audio(text: str, sample_rate: int = 44100, wpm: float = DEFAULT_WPM,
                   tone_freq_hz: float = DEFAULT_TONE_FREQ_HZ,
                   amplitude: float = DEFAULT_AMPLITUDE,
                   fade_ms: float = DEFAULT_FADE_MS) -> dict:
    """
    Synthesize a Morse-code audio recording of `text`.

    Args:
        text: the message to encode, e.g. "HELLO WORLD". Characters with
            no Morse representation are silently skipped (same behavior
            as morse_decoder.encode_text()).
        sample_rate: samples per second for the generated audio.
        wpm: words-per-minute speed, converted to a base timing unit via
            the same wpm_to_unit_seconds() the rest of the app uses.
        tone_freq_hz: frequency of the generated tone.
        amplitude: peak amplitude in [0.0, 1.0].
        fade_ms: rise/fall time on each tone burst, see _tone_burst().

    Returns:
        dict with:
            buffer  - AudioBuffer of the synthesized audio
            morse   - the Morse-notation string that was synthesized
                      (e.g. ".... . .-.. .-.. --- / .-- --- .-. .-.. -..")
            text    - the (uppercased) text actually encoded
    """
    morse = encode_text(text)
    unit = wpm_to_unit_seconds(wpm)

    chunks = []

    def add_tone(units: float):
        chunks.append(_tone_burst(units * unit, sample_rate, tone_freq_hz, amplitude, fade_ms))

    def add_silence(units: float):
        n = max(int(round(units * unit * sample_rate)), 1)
        chunks.append(np.zeros(n))

    words = morse.split(" / ") if morse else []
    for word_idx, word in enumerate(words):
        letters = word.split()
        for letter_idx, letter in enumerate(letters):
            for symbol_idx, symbol in enumerate(letter):
                add_tone(3.0 if symbol == "-" else 1.0)
                if symbol_idx < len(letter) - 1:
                    add_silence(1.0)  # intra-character gap
            if letter_idx < len(letters) - 1:
                add_silence(3.0)  # inter-character gap
        if word_idx < len(words) - 1:
            add_silence(7.0)  # word gap

    samples = np.concatenate(chunks) if chunks else np.zeros(0)

    buffer = AudioBuffer(samples=samples, sample_rate=sample_rate, source="synthesized")
    return {"buffer": buffer, "morse": morse, "text": text.upper()}


def save_audio_buffer_to_wav(buffer: AudioBuffer, path: str) -> None:
    """
    Write an AudioBuffer to a 16-bit PCM .wav file, using the exact
    inverse of audio_io.load_wav_file()'s normalization convention
    (float [-1.0, 1.0] <-> int16), so a file written here loads back
    through the existing loader without any special-casing.
    """
    clipped = np.clip(buffer.samples, -1.0, 1.0)
    int_samples = (clipped * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(path, buffer.sample_rate, int_samples)
