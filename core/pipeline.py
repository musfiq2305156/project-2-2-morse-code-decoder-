"""
Pipeline orchestration.

This module exists purely so the UI has ONE function to call rather than
importing and wiring together audio_io, signal_processing,
morse_state_machine, and morse_decoder itself. Keeping this glue code in
`core` (not `ui`) preserves the project rule that `ui` depends on `core`
and never the other way around, and keeps this function testable from
the command line with no PyQt6 involved.

Phase 7 added decode_samples(), a sibling to decode_wav_file() that skips
the "read from disk" step and runs the same pipeline directly on an
in-memory sample array - this is what live microphone decoding calls
repeatedly on a growing rolling buffer, and both functions share the
same underlying _decode_buffer() so file and live decoding can never
silently drift apart in behavior.

Both functions now also accept tone_freq_hz / tone_bandwidth_hz /
min_pulse_ms, added after real microphone testing showed the original
defaults (no frequency filtering, a peak-based threshold) worked on
clean test files but broke down on live audio. tone_freq_hz in
particular matters a lot in practice: it has to roughly match whatever
tone the person's actual Morse source (keyer, oscillator, practice app)
produces, or the band-pass filter will attenuate their real signal
instead of helping - so these are exposed here, not hardcoded, so the
UI can eventually let the person tune them for their own setup.
"""

from __future__ import annotations

import numpy as np

from core.audio_io import load_wav_file
from core.config import (
    AudioBuffer,
    DEFAULT_THRESHOLD_RATIO,
    DEFAULT_TONE_FREQ_HZ,
    DEFAULT_TONE_BANDWIDTH_HZ,
    MIN_PULSE_MS,
)
from core.signal_processing import analyze
from core.morse_state_machine import signal_to_morse
from core.morse_decoder import decode_morse
from core.signal_processing import detect_dominant_frequency


def _decode_buffer(buffer: AudioBuffer, window_ms: float, threshold_ratio: float,
                    tone_freq_hz: float | None, tone_bandwidth_hz: float,
                    min_pulse_ms: float, locked_threshold: float | None = None,
                    locked_unit_seconds: float | None = None) -> dict:
    """
    Shared core of both decode_wav_file() and decode_samples(): run
    Phases 2-4 on an already-constructed AudioBuffer. Not part of the
    public API by itself - always called through one of the two
    functions below so both entry points guarantee the same dict shape.

    locked_threshold / locked_unit_seconds: see analyze() and
    signal_to_morse() - passed straight through, for live callers that
    want to hold their calibration steady across repeated polls instead
    of silently recalibrating (and reclassifying already-decoded audio)
    every time.
    """
    phase2 = analyze(
        buffer,
        window_ms=window_ms,
        threshold_ratio=threshold_ratio,
        tone_freq_hz=tone_freq_hz,
        tone_bandwidth_hz=tone_bandwidth_hz,
        locked_threshold=locked_threshold,
    )
    phase3 = signal_to_morse(
        phase2["binary"], buffer.sample_rate, min_pulse_ms=min_pulse_ms,
        locked_unit_seconds=locked_unit_seconds,
    )
    text = decode_morse(phase3["morse"])

    return {
        "text": text,
        "morse": phase3["morse"],
        "unit_seconds": phase3["unit_seconds"],
        "duration": buffer.duration_seconds,
        "sample_rate": buffer.sample_rate,
        "samples": buffer.samples,
        "envelope": phase2["envelope"],
        "threshold": phase2["threshold"],
    }


def decode_wav_file(path: str, window_ms: float = 10.0,
                     threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
                     tone_freq_hz: float | None = DEFAULT_TONE_FREQ_HZ,
                     tone_bandwidth_hz: float = DEFAULT_TONE_BANDWIDTH_HZ,
                     min_pulse_ms: float = MIN_PULSE_MS) -> dict:
    """
    Run the full WAV-file -> decoded-text pipeline.

    Args:
        path: path to a .wav file.
        window_ms: envelope moving-average window, passed to Phase 2.
        threshold_ratio: tone/silence sensitivity, passed to Phase 2.
            Lower = more sensitive to quiet signals but more prone to
            false triggers on noise.
        tone_freq_hz: expected Morse tone frequency in Hz. Should match
            the actual pitch of whatever produces the tone (a keyer
            oscillator, practice app, etc.) - if it's off by more than
            ~100 Hz from the real tone, the band-pass filter will
            attenuate the real signal instead of just rejecting noise.
            Pass None to disable band-pass filtering entirely.
        tone_bandwidth_hz: width of the band-pass filter's pass band.
        min_pulse_ms: runs shorter than this are treated as noise
            glitches and merged away, see morse_state_machine.clean_runs().

    Returns:
        dict with:
            text          - final decoded text (str)
            morse         - Morse-notation string, e.g. "... --- ..." (str)
            unit_seconds  - estimated timing unit used for classification (float)
            duration      - audio duration in seconds (float)
            sample_rate   - audio sample rate in Hz (int)
            samples       - raw normalized waveform, numpy array (float64, [-1, 1])
            envelope      - smoothed amplitude envelope, numpy array (Phase 2)
            threshold     - scalar tone/silence cutoff used, float (Phase 2)

        This is intentionally a plain dict (not a custom class) so it can
        be passed across a Qt thread boundary via a signal without any
        special marshalling.

    Raises:
        Any exception raised by the underlying stages (e.g. FileNotFoundError,
        ValueError for a bad/empty WAV) propagates unchanged - the caller
        (the UI's worker thread) is responsible for catching and displaying it.
    """
    buffer = load_wav_file(path)
    return _decode_buffer(
        buffer,
        window_ms=window_ms,
        threshold_ratio=threshold_ratio,
        tone_freq_hz=tone_freq_hz,
        tone_bandwidth_hz=tone_bandwidth_hz,
        min_pulse_ms=min_pulse_ms,
    )


def decode_samples(samples: np.ndarray, sample_rate: int, window_ms: float = 10.0,
                    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
                    tone_freq_hz: float | None = DEFAULT_TONE_FREQ_HZ,
                    tone_bandwidth_hz: float = DEFAULT_TONE_BANDWIDTH_HZ,
                    min_pulse_ms: float = MIN_PULSE_MS,
                    source: str = "microphone",
                    locked_threshold: float | None = None,
                    locked_unit_seconds: float | None = None) -> dict:
    """
    Run the same pipeline as decode_wav_file(), but on an in-memory
    sample array instead of a file path. This is what the live
    microphone worker calls repeatedly on its rolling audio buffer.

    Args: see decode_wav_file() for window_ms/threshold_ratio/
        tone_freq_hz/tone_bandwidth_hz/min_pulse_ms - identical meaning.
        samples: normalized mono waveform, float array in [-1.0, 1.0].
        sample_rate: samples per second.
        source: human-readable label stored on the AudioBuffer (not used
            for processing, just useful context if this ever gets logged
            or debugged).
        locked_threshold / locked_unit_seconds: hold calibration steady
            across repeated polls on a growing buffer instead of
            silently recalibrating (and reclassifying already-decoded
            audio) every time - see analyze() and signal_to_morse().
            The live worker computes these itself once it has enough
            audio to calibrate confidently, then passes them back in on
            every later call.

    Returns:
        Same dict shape as decode_wav_file() - see its docstring.
    """
    buffer = AudioBuffer(
        samples=np.asarray(samples, dtype=np.float64),
        sample_rate=int(sample_rate),
        source=source,
    )
    return _decode_buffer(
        buffer,
        window_ms=window_ms,
        threshold_ratio=threshold_ratio,
        tone_freq_hz=tone_freq_hz,
        tone_bandwidth_hz=tone_bandwidth_hz,
        min_pulse_ms=min_pulse_ms,
        locked_threshold=locked_threshold,
        locked_unit_seconds=locked_unit_seconds,
    )


def detect_tone_frequency_from_wav(path: str) -> dict:
    """
    Analyze a .wav file to find its actual dominant tone frequency, for
    populating the UI's "Tone Frequency" control instead of guessing.
    See signal_processing.detect_dominant_frequency() for how this works
    and what the returned dict contains.
    """
    buffer = load_wav_file(path)
    return detect_dominant_frequency(buffer)


def detect_tone_frequency_from_samples(samples: np.ndarray, sample_rate: int) -> dict:
    """
    Same as detect_tone_frequency_from_wav(), but for an in-memory
    sample array - used to analyze a live microphone session's current
    rolling buffer on demand.
    """
    buffer = AudioBuffer(samples=np.asarray(samples, dtype=np.float64),
                          sample_rate=int(sample_rate), source="detect")
    return detect_dominant_frequency(buffer)
