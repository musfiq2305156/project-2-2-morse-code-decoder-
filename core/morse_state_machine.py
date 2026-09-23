"""
Morse timing state machine.

Phase 3 scope: take the binary tone/silence signal from Phase 2 and turn
it into standard Morse notation (dots, dashes, letter spaces, word
spaces), based on measured pulse durations and an adaptively-estimated
timing unit. Phase 4 then maps that notation to actual text.

Pipeline implemented here:
    binary tone/silence array
        -> run-length encode into (duration_seconds, is_tone) segments
        -> trim leading/trailing silence (not part of the message)
        -> estimate the base timing unit from the "on" pulse durations
        -> classify each run against unit-relative thresholds:
             on-run  -> '.' (dot) or '-' (dash)
             off-run -> '' (intra-char, no separator needed),
                        ' ' (inter-character gap), or
                        '/' (word gap)
        -> assemble into a standard Morse string, e.g. "... --- ..."

This is the differential/difference-equation-flavoured part of the
syllabus in practice: we're doing discrete-time state tracking over a
sampled signal, not continuous calculus, but the underlying idea -
describing a system by how its current sample relates to a run of past
samples - is the same discrete-time analysis mindset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from core.config import (
    DOT_DASH_BOUNDARY_RATIO,
    CHAR_GAP_BOUNDARY_RATIO,
    WORD_GAP_BOUNDARY_RATIO,
    wpm_to_unit_seconds,
    DEFAULT_WPM,
    MIN_PULSE_MS,
)


@dataclass
class Run:
    """One contiguous stretch of tone or silence."""
    duration_seconds: float
    is_tone: bool


def encode_runs(binary: np.ndarray, sample_rate: int) -> List[Run]:
    """
    Run-length encode a boolean tone/silence array into a list of Runs.

    Args:
        binary: boolean numpy array (True = tone), e.g. from
            signal_processing.binarize().
        sample_rate: samples per second, used to convert run lengths
            (in samples) to seconds.

    Returns:
        List of Run objects in chronological order. Empty if `binary`
        is empty.
    """
    if binary.size == 0:
        return []

    runs: List[Run] = []
    current_value = bool(binary[0])
    run_length = 1

    for sample in binary[1:]:
        sample = bool(sample)
        if sample == current_value:
            run_length += 1
        else:
            runs.append(Run(run_length / sample_rate, current_value))
            current_value = sample
            run_length = 1

    runs.append(Run(run_length / sample_rate, current_value))
    return runs


def clean_runs(runs: List[Run], min_seconds: float) -> List[Run]:
    """
    Merge out spurious very-short runs before any timing analysis happens.

    Real microphone audio isn't a clean square wave: background noise
    crossing the detection threshold for a few milliseconds creates tiny
    stray runs that don't correspond to anything the person actually
    keyed. Left in, these fragment what should be one clean dash into
    several short ones and corrupt the gaps around them - exactly the
    "T T TT" style garbling seen on real mic input that a clean synthetic
    test file never triggers (a synthetic tone has no noise to create
    these glitches in the first place).

    Any run shorter than `min_seconds` is folded into a neighboring run
    (added to whichever neighbor is longer, since that's more likely to
    represent the real underlying state the glitch interrupted) rather
    than simply deleted, so total duration is conserved. After folding,
    any newly-adjacent runs of the same type (which happens whenever a
    glitch between two "off" runs, say, gets removed) are merged together.

    Args:
        runs: list of Run, as produced by encode_runs(). Should be called
            before trim_silence()/estimate_unit_seconds() so the timing
            estimate isn't skewed by noise glitches.
        min_seconds: runs shorter than this are treated as noise and
            folded away. Should be comfortably shorter than the fastest
            real dot you expect (e.g. 20ms default, vs. a ~60ms dot at
            20 WPM), so real dots are never at risk of being merged away.

    Returns:
        A new list of Run objects with short glitches removed.
    """
    if not runs or min_seconds <= 0 or len(runs) <= 1:
        return list(runs)

    cleaned = list(runs)
    changed = True
    while changed and len(cleaned) > 1:
        changed = False
        for i, run in enumerate(cleaned):
            if run.duration_seconds >= min_seconds:
                continue

            if i == 0:
                nxt = cleaned[i + 1]
                cleaned[i + 1] = Run(nxt.duration_seconds + run.duration_seconds, nxt.is_tone)
                del cleaned[i]
            elif i == len(cleaned) - 1:
                prev = cleaned[i - 1]
                cleaned[i - 1] = Run(prev.duration_seconds + run.duration_seconds, prev.is_tone)
                del cleaned[i]
            else:
                prev, nxt = cleaned[i - 1], cleaned[i + 1]
                # Fold the glitch into whichever neighbor is longer, so a
                # brief noise blip doesn't arbitrarily tip a borderline
                # dot/dash or gap classification one way or the other.
                if prev.duration_seconds >= nxt.duration_seconds:
                    cleaned[i - 1] = Run(prev.duration_seconds + run.duration_seconds, prev.is_tone)
                else:
                    cleaned[i + 1] = Run(nxt.duration_seconds + run.duration_seconds, nxt.is_tone)
                del cleaned[i]

            changed = True
            break  # list mutated - restart the scan

    # Folding a glitch can leave two same-type runs newly adjacent
    # (e.g. off-run, tiny glitch tone-run, off-run -> after folding the
    # glitch into one side, two off-runs may now sit next to each other).
    # Combine those into one run so run-length invariants stay correct.
    merged: List[Run] = []
    for run in cleaned:
        if merged and merged[-1].is_tone == run.is_tone:
            merged[-1] = Run(merged[-1].duration_seconds + run.duration_seconds, merged[-1].is_tone)
        else:
            merged.append(run)
    return merged


def trim_silence(runs: List[Run]) -> List[Run]:
    """
    Drop a leading and/or trailing silence run - that's dead air before
    the first signal and after the last one, not part of the message and
    not useful for timing analysis (it's often much longer than any real
    word gap).
    """
    if not runs:
        return runs
    trimmed = list(runs)
    if trimmed and not trimmed[0].is_tone:
        trimmed = trimmed[1:]
    if trimmed and not trimmed[-1].is_tone:
        trimmed = trimmed[:-1]
    return trimmed


def estimate_unit_seconds(runs: List[Run], fallback_wpm: float = DEFAULT_WPM) -> float:
    """
    Estimate the base Morse timing unit (the duration of one dot) from
    the "on" (tone) run durations.

    Strategy: dots and dashes differ by a factor of 3, so if both are
    present in the recording there should be a clear jump when the "on"
    durations are sorted. We look for the single largest multiplicative
    gap in the sorted list; everything at/below that gap is treated as
    the "dot cluster" and its mean becomes the unit estimate. If no such
    gap exists (e.g. the message happens to be all dots, or all dashes),
    we fall back to the shortest "on" duration, since a dot is more
    likely to have been sent than assuming every symbol was a dash.

    Robustness against noise glitches: a single leftover glitch that
    barely survives debouncing could otherwise get isolated as its own
    tiny "dot cluster" (just that one point), anchoring the unit
    estimate at a bogus microsecond-scale value and corrupting every
    real dot/dash classification downstream. We guard against this by
    checking the *size* of the cluster a split would isolate: a
    genuine dot cluster should make up a reasonable share of the pulses
    seen, not just one or two stray points. If a split isolates only a
    tiny minority, those points are dropped as likely glitches and the
    search retries on what remains.

    This is checked by relative cluster size, not by comparing raw
    durations against the overall median - an earlier version of this
    function did the latter, which works fine when dots outnumber
    dashes but silently breaks otherwise: with more dashes than dots
    (easy to hit - digits and letters like O/W/G/M/K/Y are dash-heavy),
    the median itself shifts into dash-length territory, and the
    *entire real dot cluster* would get rejected as "outliers" instead
    of just the glitches. Sizing the check relative to the cluster
    itself, rather than to a global statistic that shifts with the
    dot/dash mix, avoids that failure mode regardless of which symbol
    happens to be more common in a given message.

    If there are no "on" runs at all, we fall back to a unit derived
    from `fallback_wpm` using the standard PARIS timing formula, so the
    function always returns something usable.

    Args:
        runs: list of Run (ideally already trimmed of edge silence).
        fallback_wpm: used only if there are zero tone runs.

    Returns:
        Estimated unit duration in seconds.
    """
    on_durations = sorted(run.duration_seconds for run in runs if run.is_tone)

    if not on_durations:
        return wpm_to_unit_seconds(fallback_wpm)

    if len(on_durations) == 1:
        return on_durations[0]

    durations = on_durations

    # Up to two passes: search for the dot/dash split; if the low
    # cluster it finds is too small a share of the data to trust as a
    # real dot population, drop it as noise and search again on what's
    # left. Bounded at two passes since each pass strictly shrinks the
    # data, and two is enough to recover from a small number of glitches
    # without risking discarding a message that's genuinely almost all
    # dashes.
    for _ in range(2):
        best_gap_ratio = 1.0
        best_split_index = None  # dot cluster is durations[:split_index + 1]

        for i in range(len(durations) - 1):
            a, b = durations[i], durations[i + 1]
            if a <= 0:
                continue
            ratio = b / a
            if ratio > best_gap_ratio:
                best_gap_ratio = ratio
                best_split_index = i

        # A real dot/dash split should look like roughly a 3x jump;
        # require a moderate ratio so we don't split on ordinary
        # noise/jitter.
        if best_split_index is None or best_gap_ratio <= 1.8:
            return durations[0]  # no clear split -> shortest pulse is our best guess at a dot

        cluster_size = best_split_index + 1
        min_expected = max(2, round(len(durations) * 0.15))

        if cluster_size >= min_expected or cluster_size == len(durations):
            dot_cluster = durations[:cluster_size]
            return sum(dot_cluster) / len(dot_cluster)

        # This low cluster is too small a share to trust as real dots -
        # treat it as noise glitches, drop it, and retry on the rest.
        durations = durations[cluster_size:]
        if len(durations) <= 1:
            break

    return durations[0]


def classify_runs(runs: List[Run], unit_seconds: float) -> str:
    """
    Convert a list of Runs into a standard Morse-notation string using
    thresholds relative to `unit_seconds`:

        on-run:
            duration <= DOT_DASH_BOUNDARY_RATIO * unit  -> '.'
            else                                        -> '-'
        off-run:
            duration <= CHAR_GAP_BOUNDARY_RATIO * unit   -> (intra-char, no symbol)
            duration <= WORD_GAP_BOUNDARY_RATIO * unit   -> ' '  (letter boundary)
            else                                         -> '/' (word boundary)

    Args:
        runs: list of Run, ideally pre-trimmed of leading/trailing silence.
        unit_seconds: base timing unit, e.g. from estimate_unit_seconds().

    Returns:
        A Morse-notation string, e.g. "... --- ..." for SOS. Guaranteed
        not to start/end with a stray separator even if the input had
        irregular edge timing.
    """
    if unit_seconds <= 0:
        raise ValueError("unit_seconds must be positive")

    symbols: List[str] = []
    for run in runs:
        if run.is_tone:
            symbols.append("." if run.duration_seconds <= DOT_DASH_BOUNDARY_RATIO * unit_seconds else "-")
        else:
            if run.duration_seconds <= CHAR_GAP_BOUNDARY_RATIO * unit_seconds:
                continue  # intra-character gap: no separator needed
            elif run.duration_seconds <= WORD_GAP_BOUNDARY_RATIO * unit_seconds:
                symbols.append(" ")
            else:
                symbols.append("/")

    morse = "".join(symbols)
    # Guard against edge cases producing a leading/trailing/duplicate separator.
    morse = " ".join(part for part in morse.replace("/", " / ").split(" ") if part != "")
    return morse


def signal_to_morse(binary: np.ndarray, sample_rate: int,
                     fallback_wpm: float = DEFAULT_WPM,
                     min_pulse_ms: float = MIN_PULSE_MS,
                     locked_unit_seconds: float | None = None) -> dict:
    """
    Convenience wrapper running the full Phase 3 pipeline in one call.
    This is the function Phase 4 (and later, the UI) will import.

    Args:
        binary: boolean tone/silence array, from signal_processing.binarize().
        sample_rate: samples per second.
        fallback_wpm: used only if the signal contains no tone at all,
            see estimate_unit_seconds().
        min_pulse_ms: runs shorter than this are treated as noise
            glitches and merged away before any timing analysis, see
            clean_runs(). Pass 0 to disable (useful for tests that want
            to inspect raw, unfiltered run timing).
        locked_unit_seconds: if given, use this exact timing unit
            instead of recomputing one from the current runs. Matters
            for live/streaming use for the same reason as `analyze()`'s
            `locked_threshold`: recalibrating the unit from scratch on
            every poll means it drifts as more audio arrives, which
            reclassifies already-decoded dots/dashes retroactively.
            Live callers should calibrate once early on, then hold that
            value steady for the rest of the session.

    Returns a dict with:
        morse         - Morse-notation string (e.g. "... --- ...")
        unit_seconds  - estimated timing unit used for classification
        runs          - the cleaned + trimmed list of Run objects (for debugging/plots)
    """
    runs = encode_runs(binary, sample_rate)
    runs = clean_runs(runs, min_seconds=min_pulse_ms / 1000.0)
    runs = trim_silence(runs)
    unit_seconds = locked_unit_seconds if locked_unit_seconds is not None else estimate_unit_seconds(runs, fallback_wpm=fallback_wpm)
    morse = classify_runs(runs, unit_seconds)
    return {
        "morse": morse,
        "unit_seconds": unit_seconds,
        "runs": runs,
    }
