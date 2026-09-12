"""
Background worker for running core.pipeline off the UI thread.

Decoding a WAV file is cheap for short test tones, but Phase 7 will add
real-time microphone streaming and longer recordings will take real
processing time. Establishing the QThread + signals pattern now (Phase 5)
means every later phase that needs "do core work, then update the UI"
reuses the same shape instead of ad-hoc threading being bolted on later.

Rule reminder: this file lives in `ui/` because it imports PyQt6, but the
actual work it does (core.pipeline.decode_wav_file) is 100% plain Python/
numpy - PyQt6 never leaks into `core`.
"""

from PyQt6.QtCore import QThread, pyqtSignal

from core.pipeline import decode_wav_file
from core.config import DEFAULT_THRESHOLD_RATIO, DEFAULT_TONE_FREQ_HZ, DEFAULT_TONE_BANDWIDTH_HZ, MIN_PULSE_MS


class DecodeWorker(QThread):
    """
    Runs decode_wav_file() on a background thread.

    Signals:
        finished(dict): emitted with the pipeline result on success.
        failed(str): emitted with a human-readable error message on failure.
    """

    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, wav_path: str, window_ms: float = 10.0,
                 threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
                 tone_freq_hz: float | None = DEFAULT_TONE_FREQ_HZ,
                 tone_bandwidth_hz: float = DEFAULT_TONE_BANDWIDTH_HZ,
                 min_pulse_ms: float = MIN_PULSE_MS):
        super().__init__()
        self.wav_path = wav_path
        self.window_ms = window_ms
        self.threshold_ratio = threshold_ratio
        self.tone_freq_hz = tone_freq_hz
        self.tone_bandwidth_hz = tone_bandwidth_hz
        self.min_pulse_ms = min_pulse_ms

    def run(self) -> None:
        try:
            result = decode_wav_file(
                self.wav_path,
                window_ms=self.window_ms,
                threshold_ratio=self.threshold_ratio,
                tone_freq_hz=self.tone_freq_hz,
                tone_bandwidth_hz=self.tone_bandwidth_hz,
                min_pulse_ms=self.min_pulse_ms,
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any
            # failure here (bad file, empty audio, unsupported format,
            # decoding edge case) should reach the UI as a message, not
            # crash the background thread silently.
            self.failed.emit(str(exc))
