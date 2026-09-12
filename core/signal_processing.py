"""
Signal amplitude analysis.

Phase 2 scope: turn a raw audio waveform into a binary tone/silence
signal, which Phase 3 will then walk to classify dots/dashes/gaps.

Pipeline implemented here:
    raw waveform
        -> rectify (abs value)
        -> low-pass filter / moving average (smooths rectified signal
           into an amplitude envelope, removing high-frequency ripple
           from the underlying tone itself)
        -> adaptive threshold (fraction of the envelope's peak, so it
           self-calibrates to quiet vs loud recordings)
        -> binary signal (True = tone present, False = silence)

This is a convolution-based moving average (LTI system: rectangular
kernel), which is why the syllabus's convolution/LTI section applies
directly here even before any dedicated filter design is done.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt
from scipy import ndimage

from core.config import (
    AudioBuffer,
    DEFAULT_THRESHOLD_RATIO,
    DEFAULT_TONE_FREQ_HZ,
    DEFAULT_TONE_BANDWIDTH_HZ,
)


def bandpass_filter(buffer: AudioBuffer, center_hz: float, bandwidth_hz: float,
                     order: int = 4) -> np.ndarray:
    """
    Band-pass filter the raw waveform around the expected CW tone
    frequency before any envelope/threshold work happens.

    Why this matters: a Morse tone (from a keyer, oscillator, or
    practice app) is close to a pure sine wave at one frequency
    (commonly ~600-800 Hz). Room noise, mic self-noise, voice, and hum
    are broadband or at other frequencies. Without this filter, the
    envelope detector reacts to *any* loud sound, not just the actual
    tone - which is why real microphone input was producing false
    tone detection during silence and fragmented, garbled decodes
    during real signal (see the Phase 7 follow-up discussion). Isolating
    the known tone frequency band first removes most of that
    interference before it ever reaches the threshold step.

    Args:
        buffer: input AudioBuffer.
        center_hz: expected tone frequency, e.g. 700.0.
        bandwidth_hz: total width of the pass band around center_hz,
            e.g. 200.0 means passing [center - 100, center + 100].
        order: Butterworth filter order (higher = sharper cutoff, but
            more prone to instability/ringing; 4 is a safe default).

    Returns:
        Filtered waveform, same length as buffer.samples.
    """
    nyquist = buffer.sample_rate / 2.0
    low = max(1.0, center_hz - bandwidth_hz / 2.0) / nyquist
    high = min(nyquist - 1.0, center_hz + bandwidth_hz / 2.0) / nyquist
    low = min(low, high - 1e-6)  # guard against a degenerate band

    b, a = butter(order, [low, high], btype="band")
    # filtfilt applies the filter forward then backward, giving zero
    # phase shift - important here because we care about precise pulse
    # *timing*, and a normal single-pass filter would shift edges.
    return filtfilt(b, a, buffer.samples)


def compute_envelope(samples: np.ndarray, sample_rate: int, window_ms: float = 10.0) -> np.ndarray:
    """
    Full-wave rectify the signal, then smooth it with a moving-average
    (rectangular) filter to obtain an amplitude envelope.

    Args:
        samples: waveform to envelope-detect. This is normally the
            band-pass filtered signal (see bandpass_filter above), not
            the raw waveform - filtering first means the envelope only
            reflects energy near the expected tone frequency.
        sample_rate: samples per second.
        window_ms: moving-average window length in milliseconds. Larger
            values smooth more aggressively but blur short pulses; this
            should stay well under the shortest expected Morse "dot"
            duration or dots will get smeared into neighbouring gaps.

    Returns:
        1-D numpy array, same length as `samples`, containing the
        smoothed envelope (non-negative).
    """
    rectified = np.abs(samples)

    window_samples = max(1, int(sample_rate * window_ms / 1000.0))
    kernel = np.ones(window_samples) / window_samples

    # 'same' mode keeps the envelope the same length as the input so it
    # stays sample-aligned with the original waveform for plotting/timing.
    #
    # Benchmarked this against scipy's FFT-based convolution before
    # picking one, since Phase 7's live microphone mode re-runs this
    # repeatedly on a growing rolling buffer and speed matters there.
    # For this kernel size (~a few hundred samples) and buffer lengths
    # up to a minute, direct np.convolve was consistently faster than
    # fftconvolve (its FFT setup/teardown overhead dominates at this
    # scale) - so plain direct convolution is what's used, not because
    # it's the "obvious" choice but because it measured faster.
    envelope = np.convolve(rectified, kernel, mode="same")
    return envelope


def adaptive_threshold(envelope: np.ndarray, ratio: float = DEFAULT_THRESHOLD_RATIO) -> float:
    """
    Compute a scalar threshold that separates "tone" from "silence" in
    an envelope, robust to continuous background noise.

    History: the first version of this function used `peak * ratio` (a
    fraction of the single highest envelope value). That works on a
    clean synthetic test tone - the envelope there is essentially
    bimodal, near-zero during silence and one consistent height during
    tone. It failed badly on real microphone audio: continuous
    background noise (room hum, mic self-noise) gives the envelope a
    nonzero, fairly uniform "noise floor" instead of true silence, and a
    fraction-of-*peak* threshold doesn't account for that floor at all -
    on noise-only audio the peak and the typical level end up close
    together, so the threshold could sit *below* most of the noise and
    misclassify nearly the entire buffer as one continuous tone (this is
    exactly the "detects a signal out of nowhere" symptom real
    microphone testing turned up). A percentile-ratio version
    (floor + ratio*(peak-floor)) helped but still wasn't enough: with
    a moving-average-smoothed noise envelope, ordinary random
    fluctuations routinely drift well above their own 10th-percentile
    "floor" for tens of milliseconds at a time, purely by chance -
    plenty long enough to register as a spurious dot.

    This version instead treats detection as a statistics problem: it
    estimates the noise floor's *distribution* (mean and standard
    deviation, from the quietest 20% of the envelope, which is a safe
    proxy for "probably just noise" even in a buffer that also contains
    real tone elsewhere) and sets the threshold several standard
    deviations above that mean. This is the same idea used in radar/
    signal-detection theory (a constant-false-alarm-rate style
    detector): a threshold defined in terms of "how many sigmas above
    typical noise" gives a predictable, low false-alarm rate regardless
    of the absolute noise level, which a fixed fraction-of-peak never
    could.

    Args:
        envelope: output of compute_envelope().
        ratio: sensitivity control in roughly [0.05, 0.6]. Lower values
            are more sensitive (fewer standard deviations required above
            the noise floor -> catches quieter signals, but more prone
            to false triggers on noise). Internally mapped to a
            sigma-multiplier of `3 + ratio * 12` (e.g. ratio=0.3 -> ~6.6
            sigma above the noise floor).

    Returns:
        Threshold value in the same units as `envelope`.
    """
    if envelope.size == 0:
        return 0.0

    # Percentile-range statistics, computed from the whole envelope:
    #   p_low  -> a robust "typical low" level (silence/gaps, or the
    #             noise floor if there's no real tone at all)
    #   p_high -> a robust "typical high" level (tone-on), without being
    #             thrown off by a single extreme outlier the way a raw
    #             max() would be
    # Two earlier approaches were tried and rejected before this one:
    #   1. peak-fraction of raw max(): fails on pure noise, because a
    #      long buffer of noise inevitably contains a few extreme-value
    #      envelope samples (by ordinary statistics: the more samples,
    #      the further the single highest one drifts from "typical"),
    #      and a fraction-of-that-max threshold ends up sitting inside
    #      the noise's own everyday range, not above it.
    #   2. median + MAD (median absolute deviation): robust up to a 50%
    #      "contamination" ratio in theory, but real Morse duty cycles
    #      (tone-on time as a fraction of the message) regularly sit
    #      right around 50%, so on some real messages the median itself
    #      lands inside the "tone" cluster instead of the "silence"
    #      cluster, breaking the whole estimate and rejecting real signal.
    # Percentiles fixed away from the exact median (here, 15th/85th)
    # avoid both failure modes: robust to a handful of extreme noise
    # samples like max() would not be, but tolerant of duty cycles
    # noticeably above or below 50% unlike median/MAD.
    p_low = float(np.percentile(envelope, 15))
    p_high = float(np.percentile(envelope, 85))
    peak = float(np.max(envelope))

    threshold = p_low + ratio * (p_high - p_low)

    if peak <= threshold:
        # Nothing in this buffer rises meaningfully above its own
        # typical range - i.e. no real signal (this is what a pure
        # ambient-noise buffer looks like). Set the threshold just above
        # the buffer's own peak instead, so it's all classified as
        # silence rather than leaking through via too-low a cutoff.
        return peak + 1e-9 if peak > 0 else 1e-9

    return threshold


def binarize(envelope: np.ndarray, threshold: float, hysteresis_ratio: float = 0.6) -> np.ndarray:
    """
    Apply a threshold to an envelope to produce a boolean tone/silence
    signal, using hysteresis (two thresholds instead of one) to resist
    chattering when the envelope hovers right around the cutoff.

    A single fixed threshold has a well-known problem: if the envelope
    sits close to that value for a while - exactly what a noisy live
    microphone envelope looks like - ordinary small fluctuations cause
    it to cross back and forth rapidly, producing a flurry of very short
    on/off runs instead of one clean state. This is visible directly in
    real recordings as an envelope that flickers above and below the
    threshold line continuously, even during a stretch that should read
    as pure silence.

    Hysteresis (the same principle as a Schmitt trigger in electronics,
    and the same technique Canny edge detection uses to clean up weak
    edges) fixes this by requiring a bigger rise to turn "on" than the
    drop needed to turn back "off": a contiguous stretch where the
    envelope stays at/above a lower threshold is classified as tone only
    if that stretch contains at least one point that reaches the full
    (higher) threshold somewhere in it. Ordinary noise wiggling near the
    boundary, never actually reaching the higher threshold, stays
    classified as silence throughout - it can't flicker the state on its
    own.

    Implemented with scipy.ndimage.label rather than a per-sample Python
    loop: hysteresis is inherently a "does this region contain a strong
    point" question, which connected-component labeling answers in one
    vectorized pass. A naive sample-by-sample loop would be far too slow
    to re-run several times a second on a multi-second live buffer.

    Args:
        envelope: output of compute_envelope().
        threshold: the upper (strong) cutoff, from adaptive_threshold().
        hysteresis_ratio: the lower (weak) cutoff as a fraction of
            `threshold` (0 < ratio < 1). Lower values create a wider
            gap between the two thresholds (more resistant to chatter,
            but slower to register a real, quieter state change); 1.0
            disables hysteresis entirely (identical to the old
            single-threshold behavior).

    Returns:
        Boolean array, same length as `envelope`: True = tone, False = silence.
    """
    if envelope.size == 0:
        return np.zeros(0, dtype=bool)

    low_threshold = threshold * hysteresis_ratio
    low_mask = envelope >= low_threshold
    high_mask = envelope >= threshold

    labeled, num_features = ndimage.label(low_mask)
    if num_features == 0:
        return np.zeros(len(envelope), dtype=bool)

    # For each contiguous "weak" region, does it contain any "strong" point?
    component_has_strong_point = ndimage.sum(high_mask, labeled, index=np.arange(1, num_features + 1)) > 0
    # Prepend a False for label 0 (background, i.e. below even the low
    # threshold) so this can be indexed directly by the label map.
    lookup = np.concatenate(([False], component_has_strong_point))
    return lookup[labeled]


def analyze(buffer: AudioBuffer, window_ms: float = 10.0,
            threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
            tone_freq_hz: float | None = DEFAULT_TONE_FREQ_HZ,
            tone_bandwidth_hz: float = DEFAULT_TONE_BANDWIDTH_HZ,
            locked_threshold: float | None = None) -> dict:
    """
    Convenience wrapper running the full Phase 2 pipeline in one call.
    This is the function later phases (state machine, UI) will import.

    Args:
        buffer: input AudioBuffer.
        window_ms: envelope moving-average window, see compute_envelope().
        threshold_ratio: tone/silence threshold, see adaptive_threshold().
        tone_freq_hz: expected Morse tone frequency in Hz for band-pass
            filtering (see bandpass_filter()). Pass None to skip
            filtering entirely and envelope-detect the raw waveform -
            this was the only behavior before real microphone testing
            showed it isn't robust enough for noisy live audio, but it's
            kept available since a clean pre-recorded file doesn't need
            it and skipping it is slightly cheaper.
        tone_bandwidth_hz: width of the band-pass filter's pass band.
        locked_threshold: if given, use this exact value instead of
            recomputing one from the current envelope. This matters for
            live/streaming use: recomputing the threshold from scratch
            on every poll means it drifts slightly as more audio is
            captured, which silently reclassifies *already-decoded*
            portions of the buffer, not just new audio - visible to the
            user as earlier letters flickering/changing even though
            that part of the recording never changed. Live callers
            should compute a threshold once early on, then pass it back
            in on every subsequent call to keep classification stable.

    Returns:
        dict with:
            envelope   - smoothed amplitude envelope (np.ndarray)
            threshold  - scalar cutoff used (float)
            binary     - boolean tone/silence array (np.ndarray)
            filtered   - the band-pass filtered waveform actually used
                         for envelope detection (np.ndarray); equals
                         buffer.samples unchanged if tone_freq_hz is None
    """
    if tone_freq_hz is not None:
        filtered = bandpass_filter(buffer, center_hz=tone_freq_hz, bandwidth_hz=tone_bandwidth_hz)
    else:
        filtered = buffer.samples

    envelope = compute_envelope(filtered, buffer.sample_rate, window_ms=window_ms)
    threshold = locked_threshold if locked_threshold is not None else adaptive_threshold(envelope, ratio=threshold_ratio)
    binary = binarize(envelope, threshold)
    return {
        "envelope": envelope,
        "threshold": threshold,
        "binary": binary,
        "filtered": filtered,
    }


def detect_dominant_frequency(buffer: AudioBuffer, freq_range: tuple[float, float] = (300.0, 2000.0)) -> dict:
    """
    Estimate the actual tone frequency present in a recording.

    Why this exists: the band-pass filter in bandpass_filter() only
    helps if its center frequency roughly matches the *real* tone being
    sent. Guessing wrong (e.g. assuming 700 Hz when the actual source
    produces 900 Hz) means the filter mostly rejects the real signal and
    passes through leftover broadband noise instead - which shows up as
    a continuously jagged, noisy envelope with no clean on/off structure,
    even though a real tone is genuinely present in the recording. This
    function finds the actual frequency directly from the audio instead
    of assuming one, using a Fourier transform (the Phase 4-adjacent
    "frequency domain analysis" part of the DSP syllabus, applied here
    practically instead of just theoretically).

    Approach: a plain FFT over the whole buffer would dilute the tone's
    spectral peak with energy from silent/noisy stretches. Instead, this
    only analyzes the loudest half of the recording by amplitude (a
    simple energy mask - no band-pass filtering yet, since we don't know
    the frequency to filter around), which is a reasonable proxy for
    "where the tone is actually playing" as long as the tone's duty
    cycle is under ~50%, then takes the FFT of that masked signal and
    reports the strongest frequency within a plausible CW tone range.

    Args:
        buffer: input AudioBuffer (raw, unfiltered).
        freq_range: (low_hz, high_hz) - only frequencies in this range
            are considered candidates, to ignore very low rumble/hum and
            very high hiss that are never real CW tones.

    Returns:
        dict with:
            frequency_hz  - the detected dominant frequency (float)
            confidence    - ratio of the peak's magnitude to the median
                             magnitude in the search range (float); much
                             greater than 1 means a clear, distinct tone
                             was found, close to 1 means the spectrum is
                             fairly flat (probably no real tone present,
                             e.g. a noise-only or silent recording)
    """
    samples = buffer.samples
    if samples.size == 0:
        return {"frequency_hz": DEFAULT_TONE_FREQ_HZ, "confidence": 0.0}

    # Energy mask: keep only the loudest half of the recording (by a
    # coarse envelope), zeroing out the rest so quiet/noisy stretches
    # don't dilute the tone's spectral peak.
    window_samples = max(1, int(buffer.sample_rate * 0.01))
    kernel = np.ones(window_samples) / window_samples
    coarse_envelope = np.convolve(np.abs(samples), kernel, mode="same")
    mask = coarse_envelope >= np.median(coarse_envelope)
    masked = samples * mask

    windowed = masked * np.hanning(len(masked))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / buffer.sample_rate)

    in_range = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not np.any(in_range):
        return {"frequency_hz": DEFAULT_TONE_FREQ_HZ, "confidence": 0.0}

    range_spectrum = spectrum[in_range]
    range_freqs = freqs[in_range]

    peak_index = int(np.argmax(range_spectrum))
    peak_freq = float(range_freqs[peak_index])
    peak_magnitude = float(range_spectrum[peak_index])
    median_magnitude = float(np.median(range_spectrum)) or 1e-12
    confidence = peak_magnitude / median_magnitude

    return {"frequency_hz": peak_freq, "confidence": confidence}
