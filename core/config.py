"""
Shared constants and configuration for the DSP core.

Keeping these in one place means later phases (thresholding, timing
classification, decoding) all agree on units and defaults without
importing each other's internals.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Audio defaults
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 44100          # Hz, used when capturing from mic
SUPPORTED_WAV_DTYPES = ("int16", "int32", "float32", "float64")


@dataclass
class AudioBuffer:
    """
    Plain data container passed between core modules (and later, to the UI
    via Qt signals). Deliberately not a numpy subclass so it stays simple
    to serialize / inspect / unit test.

    Attributes:
        samples: 1-D numpy array of mono audio samples, normalized to
                  the range [-1.0, 1.0] as float64.
        sample_rate: samples per second (Hz).
        source: human-readable origin, e.g. "file:test.wav" or "microphone".
    """
    samples: "np.ndarray"
    sample_rate: int
    source: str

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / float(self.sample_rate)


# ---------------------------------------------------------------------------
# Placeholders for later phases (kept here now so config.py is the single
# source of truth as the project grows; unused until Phase 2/3).
# ---------------------------------------------------------------------------

DEFAULT_WPM = 20                     # words per minute, standard Morse timing
DEFAULT_THRESHOLD_RATIO = 0.3        # sensitivity: fraction of the way from
                                      # typical-low to typical-high envelope
                                      # level used as the tone/silence cutoff

# ---------------------------------------------------------------------------
# Morse timing ratios (Phase 3)
# ---------------------------------------------------------------------------
# Standard Morse timing, expressed as multiples of one "unit":
#   dot              = 1 unit (on)
#   dash             = 3 units (on)
#   intra-char gap   = 1 unit (off, between symbols of the same letter)
#   inter-char gap   = 3 units (off, between letters)
#   word gap         = 7 units (off, between words)
#
# The *_BOUNDARY_RATIO constants below are the decision thresholds used to
# classify a measured duration, placed at the midpoint (on a log-ish scale)
# between two categories so a slightly-off recording still classifies
# correctly rather than needing an exact 1x/3x/7x match.
DOT_DASH_BOUNDARY_RATIO = 2.0         # >2 units on-time -> dash, else dot
CHAR_GAP_BOUNDARY_RATIO = 2.0         # >2 units off-time -> at least a char gap
WORD_GAP_BOUNDARY_RATIO = 5.0         # >5 units off-time -> a word gap

# When a pulse train contains no clear short/long split (e.g. a single
# repeated symbol), or no tone at all, fall back to a unit length derived
# from DEFAULT_WPM using the standard PARIS timing formula.
def wpm_to_unit_seconds(wpm: float = DEFAULT_WPM) -> float:
    return 1.2 / wpm

# ---------------------------------------------------------------------------
# Live microphone mode (Phase 7)
# ---------------------------------------------------------------------------
LIVE_MAX_BUFFER_SECONDS = 30.0   # rolling window cap so live decode stays cheap
LIVE_POLL_INTERVAL_SECONDS = 0.4 # how often the live worker re-decodes and updates the UI

# ---------------------------------------------------------------------------
# Noise robustness for real microphone input (added after real-world testing
# showed the original peak-based threshold + no frequency filtering worked
# on clean test files but broke down on live mic audio - false tone
# detection during silence, and fragmented/garbled decodes during real
# signal, both caused by broadband background noise the detector had no
# way to distinguish from an actual Morse tone.)
# ---------------------------------------------------------------------------
DEFAULT_TONE_FREQ_HZ = 700.0       # typical CW sidetone / practice-oscillator pitch
DEFAULT_TONE_BANDWIDTH_HZ = 200.0  # band-pass width around the tone frequency
MIN_PULSE_MS = 30.0                # runs shorter than this are treated as noise
                                    # glitches and merged into a neighboring run
