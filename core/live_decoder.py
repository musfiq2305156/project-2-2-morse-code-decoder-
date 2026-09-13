"""
Incremental (streaming) Morse decoder for live microphone / simulated
live input.

--------------------------------------------------------------------
Why this file exists (the bug it fixes)
--------------------------------------------------------------------
The original Phase 7 live pipeline (core.pipeline.decode_samples, called
from ui.live_worker.LiveDecodeWorker) worked by re-running the *entire*
batch pipeline - band-pass filter, envelope, threshold, dot/dash timing
estimate, full decode - from scratch on a growing rolling buffer, once
per poll (every ~0.4s). That looks reasonable but every one of those
stages is a function of the *whole buffer's* statistics:

  * signal_processing.bandpass_filter() uses scipy.signal.filtfilt, a
    zero-phase filter that is explicitly non-causal - it needs samples
    from *after* a point in time to compute that point's filtered
    value correctly. On a live/growing buffer there is no "after" for
    the newest samples yet, and the correction it applies near the
    live edge keeps changing retroactively on every new poll as real
    "future" samples arrive.
  * signal_processing.adaptive_threshold() takes the 15th/85th
    percentile of the *entire* envelope. As the buffer grows, those
    percentiles shift, so the tone/silence cutoff used for audio
    received two seconds ago is silently different on every poll.
  * morse_state_machine.estimate_unit_seconds() finds the biggest gap
    in the sorted list of *all* on-durations seen so far, so the
    dot/dash boundary for an early letter can flip as later letters
    change what "sorted list" looks like.

Put together: nothing about a live decode was ever final. What you saw
on screen (and what froze when you hit Stop) was not a stable, growing
transcript - it was whichever full re-analysis of the whole buffer
happened to land on the last poll. That's the actual cause of the
garbled live/simulated output (this affects "Simulate from File" too,
since ui.main_window wires it through the exact same LiveDecodeWorker -
see its own docstring: "drives the exact same live pipeline").

--------------------------------------------------------------------
The fix
--------------------------------------------------------------------
This module is a genuine streaming decoder: audio comes in small
chunks via push(), and every stage keeps only the small amount of
state it needs to keep going, so nothing already decided is ever
revisited:

  * band-pass filtering uses scipy.signal.sosfilt with a persisted
    filter state (`zi`) carried across calls - the causal counterpart
    of filtfilt. A causal filter has a fixed processing delay, but
    that delay is constant, so it does not distort *relative* pulse
    timing (which is all the Morse decision depends on).
  * the envelope is a causal low-pass (sosfilt again, its own
    persisted `zi`) instead of the batch version's convolution over
    the whole array - it is, literally, the difference equation the
    syllabus's "time domain analysis of LTI systems" section covers,
    evaluated recursively one sample at a time.
  * the tone/silence threshold is tracked with a running noise-floor /
    signal-ceiling estimate (an asymmetric exponential moving
    average - itself a first-order difference equation) instead of a
    percentile over history that keeps shifting.
  * a run (a contiguous stretch of tone or silence) is only finalized
    - classified as a dot/dash/gap and appended to the transcript -
    once a debounce window confirms the state has genuinely changed.
    Once finalized, a run is never reclassified.

The batch, one-shot file pipeline in core.pipeline / core.signal_processing
/ core.morse_state_machine is untouched and still used for "Load WAV
File" - it works correctly and doesn't have this problem, precisely
*because* it only ever runs once, over the complete, final recording.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from core.config import (
    DOT_DASH_BOUNDARY_RATIO,
    CHAR_GAP_BOUNDARY_RATIO,
    WORD_GAP_BOUNDARY_RATIO,
    MIN_PULSE_MS,
    DEFAULT_WPM,
    DEFAULT_THRESHOLD_RATIO,
    wpm_to_unit_seconds,
)
from core.morse_decoder import decode_letter

#bandpass
def _bandpass_sos(sample_rate: float, center_hz: float, bandwidth_hz: float, order: int = 4):
    nyquist = sample_rate / 2.0
    low = max(1.0, center_hz - bandwidth_hz / 2.0) / nyquist
    high = min(nyquist - 1.0, center_hz + bandwidth_hz / 2.0) / nyquist
    low = min(low, high - 1e-6)
    return butter(order, [low, high], btype="band", output="sos")


def _lowpass_sos(sample_rate: float, cutoff_hz: float, order: int = 2):
    nyquist = sample_rate / 2.0
    normal_cutoff = min(0.99, cutoff_hz / nyquist)
    return butter(order, normal_cutoff, btype="low", output="sos")


class IncrementalMorseDecoder:
    """
    Streaming Morse decoder. Feed raw, normalized ([-1, 1]) mono audio
    chunks via push(); read .text / .morse for the transcript so far.
    Call flush() when the stream ends to commit any in-progress letter.

    Nothing this class has already appended to .text/.morse is ever
    revised by a later push() call - that's the whole point.
    """

    def __init__(self, sample_rate: int, tone_freq_hz: Optional[float] = 700.0,
                 tone_bandwidth_hz: float = 200.0,
                 threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
                 min_pulse_ms: float = MIN_PULSE_MS,
                 fallback_wpm: float = DEFAULT_WPM,
                 envelope_cutoff_hz: float = 25.0):
        self.sample_rate = sample_rate
        self.threshold_ratio = threshold_ratio
        self.min_pulse_samples = max(1, int(sample_rate * min_pulse_ms / 1000.0))

        self._bp_sos = None
        self._bp_zi = None
        if tone_freq_hz:
            self._bp_sos = _bandpass_sos(sample_rate, tone_freq_hz, tone_bandwidth_hz)
            self._bp_zi = sosfilt_zi(self._bp_sos)

        self._env_sos = _lowpass_sos(sample_rate, envelope_cutoff_hz)
        self._env_zi = sosfilt_zi(self._env_sos)

        # Running noise-floor / signal-ceiling estimate used instead of
        # a whole-history percentile - see module docstring.
        self._floor = 0.0
        self._ceiling = 1e-6
        self._floor_ready = False

        self._unit_seconds = wpm_to_unit_seconds(fallback_wpm)

        # Debounced run tracking. `confirmed_*` is the settled state
        # we're counting toward a symbol; `candidate_*` is a possible
        # state change we haven't accepted yet (it might be a noise
        # glitch and get reabsorbed into the confirmed run).
        self._confirmed_is_tone = False
        self._confirmed_samples = 0
        self._candidate_is_tone: Optional[bool] = None
        self._candidate_samples = 0

        self._letter_symbols: List[str] = []
        self._words: List[List[str]] = [[]]
        self.text = ""
        self.morse = ""

        # Kept only for the UI's waveform display - not used for decoding.
        self.last_chunk_envelope = np.zeros(0)
        self.last_threshold = 0.0

    # -- public API -----------------------------------------------------

    def push(self, chunk: np.ndarray) -> None:
        """Feed the next chunk of normalized mono audio ([-1, 1])."""
        if chunk.size == 0:
            return
        chunk = np.asarray(chunk, dtype=np.float64)

        if self._bp_sos is not None:
            filtered, self._bp_zi = sosfilt(self._bp_sos, chunk, zi=self._bp_zi)
        else:
            filtered = chunk

        rectified = np.abs(filtered)
        envelope, self._env_zi = sosfilt(self._env_sos, rectified, zi=self._env_zi)
        self.last_chunk_envelope = envelope

        self._update_floor_ceiling(envelope)
        threshold = self._current_threshold()
        self.last_threshold = threshold

        binary = envelope >= threshold
        for is_on in binary:
            self._step(bool(is_on))

    def flush(self) -> None:
        """Commit any run/letter still in progress. Call when the stream stops."""
        self._finalize_confirmed_run(force=True)
        self._commit_letter()

    # -- internals: threshold tracking -----------------------------------

    def _update_floor_ceiling(self, envelope: np.ndarray) -> None:
        lo = float(np.min(envelope))
        hi = float(np.max(envelope))
        if not self._floor_ready:
            self._floor, self._ceiling = lo, max(hi, lo + 1e-6)
            self._floor_ready = True
            return
        # Floor decays slowly toward the chunk minimum, so a chunk that's
        # entirely tone (no silence in it) can't yank the floor upward.
        self._floor += 0.05 * (lo - self._floor)
        self._floor = min(self._floor, hi)  # never let floor exceed this chunk's max
        # Ceiling jumps up immediately to a new peak, decays slowly
        # otherwise, so it reflects real signal strength.
        if hi > self._ceiling:
            self._ceiling = hi
        else:
            self._ceiling += 0.02 * (hi - self._ceiling)

    def _current_threshold(self) -> float:
        span = max(self._ceiling - self._floor, 1e-9)
        return self._floor + self.threshold_ratio * span

    # -- internals: debounced run tracking --------------------------------

    def _step(self, is_on: bool) -> None:
        if self._candidate_is_tone is None:
            if is_on == self._confirmed_is_tone:
                self._confirmed_samples += 1
            else:
                self._candidate_is_tone = is_on
                self._candidate_samples = 1
            return

        if is_on == self._candidate_is_tone:
            self._candidate_samples += 1
            if self._candidate_samples >= self.min_pulse_samples:
                # Confirmed: the state really did change. Finalize the
                # run that just ended, then start counting the new one.
                self._finalize_confirmed_run(force=False)
                self._confirmed_is_tone = self._candidate_is_tone
                self._confirmed_samples = self._candidate_samples
                self._candidate_is_tone = None
                self._candidate_samples = 0
        else:
            # Wobbled back to the confirmed state before the debounce
            # window elapsed - a noise glitch. Fold it into the
            # confirmed run's duration instead of treating it as real.
            self._confirmed_samples += self._candidate_samples + 1
            self._candidate_is_tone = None
            self._candidate_samples = 0

    def _finalize_confirmed_run(self, force: bool) -> None:
        samples = self._confirmed_samples
        is_tone = self._confirmed_is_tone
        self._confirmed_samples = 0
        if samples <= 0:
            return

        duration = samples / self.sample_rate

        if is_tone:
            symbol = "." if duration <= DOT_DASH_BOUNDARY_RATIO * self._unit_seconds else "-"
            self._letter_symbols.append(symbol)
            if symbol == ".":
                # Only dots nudge the running unit estimate - same
                # reasoning as the batch estimator (a dash is 3x a dot,
                # so treating every short pulse as the reference point
                # is more stable than averaging dots and dashes
                # together). Adapts slowly (small step) so one odd
                # measurement can't swing the timing wildly.
                self._unit_seconds += 0.2 * (duration - self._unit_seconds)
        else:
            if duration <= CHAR_GAP_BOUNDARY_RATIO * self._unit_seconds:
                return  # intra-character gap - letter isn't finished
            self._commit_letter()
            if duration > WORD_GAP_BOUNDARY_RATIO * self._unit_seconds:
                self._commit_word()

    def _commit_letter(self) -> None:
        if not self._letter_symbols:
            return
        pattern = "".join(self._letter_symbols)
        self._letter_symbols = []
        self._words[-1].append(pattern)
        self.text += decode_letter(pattern)
        self.morse = self._render_morse()

    def _commit_word(self) -> None:
        if self._words[-1]:
            self._words.append([])
            self.text += " "
            self.morse = self._render_morse()

    def _render_morse(self) -> str:
        parts = [" ".join(letters) for letters in self._words if letters]
        return " / ".join(parts)

    @property
    def unit_seconds(self) -> float:
        return self._unit_seconds
