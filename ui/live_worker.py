"""
Live decoding worker.

Phase 7 scope: manage a microphone (or simulated) audio stream on a
background thread, accumulate incoming chunks into a rolling buffer, and
periodically re-run the core pipeline on that buffer so the UI can show
a continuously-updating decode - waveform, Morse notation, and text -
without ever blocking the UI thread.

Reuses the exact same threading shape established in Phase 5's
DecodeWorker (a QThread emitting result dicts via a signal), just looped
instead of one-shot.
"""

from __future__ import annotations

import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.pipeline import decode_samples
from core.config import (
    LIVE_MAX_BUFFER_SECONDS,
    LIVE_POLL_INTERVAL_SECONDS,
    DEFAULT_THRESHOLD_RATIO,
    DEFAULT_TONE_FREQ_HZ,
    DEFAULT_TONE_BANDWIDTH_HZ,
    MIN_PULSE_MS,
)

# Once the live buffer has at least this much audio AND this many
# decoded dot/dash symbols, freeze the threshold and timing-unit
# calibration for the rest of the session. Recalibrating from scratch on
# every poll (the original Phase 7 behavior) makes the numeric threshold
# and unit estimate drift slightly as more audio arrives - which quietly
# reclassifies *already-decoded* letters, not just new ones, and shows
# up to the user as earlier parts of the message flickering/changing
# even though that portion of the recording never changed. Calibrating
# once from an initial "settling" window and holding steady afterward
# (much like how a real CW operator "gets an ear in" at the start of a
# transmission) keeps the displayed transcript stable.
CALIBRATION_MIN_SECONDS = 2.5
CALIBRATION_MIN_SYMBOLS = 5


class LiveDecodeWorker(QThread):
    """
    Runs a live audio stream (MicrophoneStream or SimulatedMicrophoneStream)
    on a background thread: starts it, polls it for new audio at a fixed
    interval, keeps a rolling buffer capped at LIVE_MAX_BUFFER_SECONDS
    (older audio is dropped so re-decoding stays cheap and the message
    being decoded reflects "recent" speech rather than growing forever),
    and emits a fresh decode result after every poll that has audio.

    Signals:
        updated(dict): emitted with the latest decode_samples() result
            after each successful poll with audio in the buffer.
        failed(str): emitted with a human-readable error message if the
            stream fails to start (e.g. no microphone available) or hits
            an unrecoverable error while running.
    """

    updated = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, mic_stream, window_ms: float = 10.0,
                 threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
                 tone_freq_hz: float | None = DEFAULT_TONE_FREQ_HZ,
                 tone_bandwidth_hz: float = DEFAULT_TONE_BANDWIDTH_HZ,
                 min_pulse_ms: float = MIN_PULSE_MS,
                 poll_interval: float = LIVE_POLL_INTERVAL_SECONDS,
                 max_buffer_seconds: float = LIVE_MAX_BUFFER_SECONDS):
        super().__init__()
        self.mic_stream = mic_stream
        self.window_ms = window_ms
        self.threshold_ratio = threshold_ratio
        self.tone_freq_hz = tone_freq_hz
        self.tone_bandwidth_hz = tone_bandwidth_hz
        self.min_pulse_ms = min_pulse_ms
        self.poll_interval = poll_interval
        self.max_buffer_seconds = max_buffer_seconds
        self._buffer = np.zeros(0, dtype=np.float32)
        self._running = False
        self._locked_threshold: float | None = None
        self._locked_unit_seconds: float | None = None

    def run(self) -> None:
        try:
            self.mic_stream.start()
        except Exception as exc:  # noqa: BLE001 - surface any stream startup
            # failure (missing PortAudio, no input device, permission
            # denied, etc.) to the UI rather than crashing the thread.
            self.failed.emit(f"Could not start audio input: {exc}")
            return

        self._running = True
        try:
            while self._running:
                time.sleep(self.poll_interval)
                if not self._running:
                    break

                new_chunk = self.mic_stream.drain()
                if new_chunk.size:
                    self._buffer = np.concatenate([self._buffer, new_chunk])
                    max_samples = int(self.max_buffer_seconds * self.mic_stream.sample_rate)
                    if len(self._buffer) > max_samples:
                        # Keep only the most recent audio - an unbounded
                        # buffer would make every re-decode slower and
                        # slower the longer a live session runs.
                        self._buffer = self._buffer[-max_samples:]

                if self._buffer.size == 0:
                    continue

                try:
                    result = decode_samples(
                        self._buffer,
                        self.mic_stream.sample_rate,
                        window_ms=self.window_ms,
                        threshold_ratio=self.threshold_ratio,
                        tone_freq_hz=self.tone_freq_hz,
                        tone_bandwidth_hz=self.tone_bandwidth_hz,
                        min_pulse_ms=self.min_pulse_ms,
                        source="microphone",
                        locked_threshold=self._locked_threshold,
                        locked_unit_seconds=self._locked_unit_seconds,
                    )

                    # Once there's enough audio and enough decoded
                    # symbols to trust the calibration, freeze it so the
                    # rest of the session stays stable instead of
                    # silently redrawing already-decoded letters.
                    if self._locked_threshold is None:
                        buffer_seconds = len(self._buffer) / self.mic_stream.sample_rate
                        symbol_count = sum(1 for ch in result["morse"] if ch in ".-")
                        if buffer_seconds >= CALIBRATION_MIN_SECONDS and symbol_count >= CALIBRATION_MIN_SYMBOLS:
                            self._locked_threshold = result["threshold"]
                            self._locked_unit_seconds = result["unit_seconds"]

                    self.updated.emit(result)
                except Exception:
                    # A single bad decode tick (e.g. a pathological buffer
                    # state) shouldn't kill the whole live session - skip
                    # this tick and try again on the next poll.
                    continue
        finally:
            self.mic_stream.stop()

    def stop(self) -> None:
        """Signal the run() loop to exit and wait for the thread to finish."""
        self._running = False
        self.wait()

    def get_buffer_snapshot(self) -> tuple[np.ndarray, int]:
        """
        Return a copy of the current rolling buffer and its sample rate,
        for on-demand analysis (e.g. the UI's "Detect Frequency" button)
        without disturbing the live decode loop itself.
        """
        return self._buffer.copy(), self.mic_stream.sample_rate
