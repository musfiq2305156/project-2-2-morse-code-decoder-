"""
Audio input handling.

Phase 1: loading pre-recorded .wav files into a normalized numpy array.

Phase 7 adds live microphone capture via sounddevice.InputStream, plus a
SimulatedMicrophoneStream that replays a .wav file chunk-by-chunk at
real-time pace behind the exact same interface. The simulated stream
exists for two reasons: it's how this module gets tested in environments
with no physical microphone (or, as discovered while building this, no
PortAudio library at all), and it doubles as a genuinely useful "demo
mode" for showing the live pipeline working without needing to actually
speak/key into a mic.

`sounddevice` is imported lazily inside MicrophoneStream, not at module
level: on a machine where PortAudio isn't installed, `import sounddevice`
raises immediately, and doing that at module import time would crash the
entire app (including WAV file decoding, which has nothing to do with
live audio) just because the live-mic feature's dependency is missing.
Deferring the import means the app still runs fine and only the
"Start Microphone" action fails, with a catchable, user-facing error.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
from scipy.io import wavfile

from core.config import AudioBuffer, SUPPORTED_WAV_DTYPES, DEFAULT_SAMPLE_RATE


def load_wav_file(path: str) -> AudioBuffer:
    """
    Load a .wav file and return it as an AudioBuffer with samples
    normalized to float64 in the range [-1.0, 1.0], collapsed to mono.

    Args:
        path: filesystem path to a .wav file.

    Returns:
        AudioBuffer with mono, normalized samples.

    Raises:
        ValueError: if the file's sample dtype isn't one we know how to
            normalize, or the file contains no samples.
        FileNotFoundError: if the path doesn't exist (raised by scipy).
    """
    sample_rate, raw_samples = wavfile.read(path)

    if raw_samples.size == 0:
        raise ValueError(f"'{path}' contains no audio samples.")

    dtype_name = raw_samples.dtype.name
    if dtype_name not in SUPPORTED_WAV_DTYPES:
        raise ValueError(
            f"Unsupported WAV sample format '{dtype_name}'. "
            f"Supported formats: {SUPPORTED_WAV_DTYPES}"
        )

    # Collapse stereo/multi-channel to mono by averaging channels.
    if raw_samples.ndim > 1:
        raw_samples = raw_samples.mean(axis=1)

    normalized = _normalize_to_float(raw_samples, dtype_name)

    return AudioBuffer(
        samples=normalized,
        sample_rate=int(sample_rate),
        source=f"file:{path}",
    )


def _normalize_to_float(samples: np.ndarray, dtype_name: str) -> np.ndarray:
    """
    Convert integer PCM samples to float64 in [-1.0, 1.0]. Float WAVs are
    assumed to already be in that range and are just cast to float64.
    """
    samples = samples.astype(np.float64)

    if dtype_name == "int16":
        return samples / np.iinfo(np.int16).max
    if dtype_name == "int32":
        return samples / np.iinfo(np.int32).max
    # float32 / float64: already normalized by convention of the WAV format
    return samples


def describe(buffer: AudioBuffer) -> str:
    """Human-readable summary of an AudioBuffer, used by the CLI test script."""
    peak = np.max(np.abs(buffer.samples)) if buffer.samples.size else 0.0
    return (
        f"source          : {buffer.source}\n"
        f"sample_rate     : {buffer.sample_rate} Hz\n"
        f"num_samples     : {len(buffer.samples)}\n"
        f"duration        : {buffer.duration_seconds:.3f} s\n"
        f"amplitude range : [{buffer.samples.min():.4f}, {buffer.samples.max():.4f}]\n"
        f"peak abs value  : {peak:.4f}"
    )


# ---------------------------------------------------------------------------
# Live audio streaming (Phase 7)
#
# Both classes below expose the same small interface so callers (the UI's
# LiveDecodeWorker) can treat a real microphone and a simulated one
# identically:
#     .sample_rate            -> int
#     .start()                -> begins producing chunks
#     .stop()                 -> stops producing chunks, releases resources
#     .drain() -> np.ndarray  -> non-blockingly returns all chunks queued
#                                 since the last call (concatenated), or
#                                 an empty array if none are available yet
# ---------------------------------------------------------------------------


class MicrophoneStream:
    """
    Captures live audio from the system's default input device into a
    thread-safe queue of small chunks, using sounddevice.InputStream.

    This class does no DSP itself - it only produces normalized float
    samples. Callers (Phase 7's LiveDecodeWorker) pull chunks out via
    drain() and feed them into signal_processing / morse_state_machine /
    morse_decoder, same as file-based decoding.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, channels: int = 1,
                 blocksize: int = 1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        # Runs on PortAudio's own audio thread, not the Qt thread - keep
        # this fast and non-blocking. indata is float32 in [-1.0, 1.0].
        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self.chunk_queue.put(mono)

    def start(self) -> None:
        """
        Open and start the input stream.

        Raises:
            Whatever sounddevice/PortAudio raises if audio input isn't
            available - e.g. OSError("PortAudio library not found") if
            PortAudio isn't installed, or a PortAudioError if there's no
            input device. Callers are expected to catch this (the UI's
            LiveDecodeWorker turns it into a `failed` signal).
        """
        import sounddevice as sd  # lazy import, see module docstring

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def drain(self) -> np.ndarray:
        """Non-blockingly pull all currently queued chunks into one array."""
        return _drain_queue(self.chunk_queue)


class SimulatedMicrophoneStream:
    """
    Drop-in replacement for MicrophoneStream that replays a pre-recorded
    .wav file chunk-by-chunk in real time, behind the identical
    start()/stop()/drain()/sample_rate interface.

    Two uses for this:
      1. Testing/CI: the live decoding pipeline (buffering, periodic
         re-decode, UI updates) can be exercised deterministically
         without any physical microphone or even a working PortAudio
         install.
      2. A "demo mode" in the app itself: showing the live view working
         (e.g. in a lecture hall, exam room, or noisy environment) by
         replaying a known-good recording instead of relying on a mic
         picking up a real Morse key cleanly.
    """

    def __init__(self, wav_path: str, blocksize: int = 1024):
        buffer = load_wav_file(wav_path)
        self.sample_rate = buffer.sample_rate
        self._samples = buffer.samples.astype(np.float32)
        self.blocksize = blocksize
        self.chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        position = 0
        n = len(self._samples)
        chunk_duration = self.blocksize / self.sample_rate
        while position < n and not self._stop_event.is_set():
            chunk = self._samples[position: position + self.blocksize]
            self.chunk_queue.put(chunk)
            position += self.blocksize
            time.sleep(chunk_duration)  # pace it to mimic real-time capture

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def drain(self) -> np.ndarray:
        return _drain_queue(self.chunk_queue)


def _drain_queue(q: "queue.Queue[np.ndarray]") -> np.ndarray:
    """Shared helper: non-blockingly collect everything currently queued."""
    chunks = []
    while True:
        try:
            chunks.append(q.get_nowait())
        except queue.Empty:
            break
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)
