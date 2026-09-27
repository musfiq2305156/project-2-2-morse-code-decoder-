# Morse Code Detector

A Python/PyQt6 desktop app that decodes Morse code from a `.wav` file or a
live microphone feed, built as a signals & systems course project. The
architecture keeps DSP (`core/`) and UI (`ui/`) strictly separate, so the
decoding pipeline can be understood, tested, and graded independently of
the GUI.

## Features

- **Signals & Systems Theory Guide**: See [`SIGNALS_THEORY.md`](SIGNALS_THEORY.md) for a detailed breakdown of how course topics (Sampling, Band-pass Filtering, Convolution, FFT, Hysteresis, State Machine) are applied.
- **Load a `.wav` file** and decode it to text.
- **Simulate live input from a file** (replays a `.wav` chunk-by-chunk in
  real time) - useful for demos and for testing the live pipeline without
  a physical microphone.
- **Decode live from a microphone**, with a rolling buffer re-decoded
  several times a second.
- **Auto-detect the tone frequency** from the actual recording (FFT-based)
  instead of guessing - press "Detect Frequency" after loading audio.
- **Adjustable Tone Frequency and Sensitivity** controls, for tuning to a
  specific recording setup.
- **Automatic recording-quality warning**: if the recording's actual
  on/off contrast is too weak for reliable decoding, the app says so and
  explains why, rather than silently producing garbled text.
- **Export decoded text** to a `.txt` file.
- **Text \u2192 Morse encoder**: type any message, see its Morse notation, and export it as an actual `.wav` audio file - the inverse of the decode pipeline.
- **Audio playback during "Simulate from File"**: hear the file being simulated while watching it decode live.
- Embedded waveform + envelope/threshold visualization for every decode.

## Architecture

```
morse_detector/
├── core/                      # Pure DSP - NO PyQt imports allowed here
│   ├── config.py               # shared constants, AudioBuffer dataclass
│   ├── audio_io.py             # WAV loading, MicrophoneStream, SimulatedMicrophoneStream
│   ├── signal_processing.py    # band-pass filter, envelope, adaptive threshold, hysteresis binarization
│   ├── morse_state_machine.py  # run-length encoding, timing calibration, dot/dash classification
│   ├── morse_decoder.py        # Morse <-> text lookup tables
│   ├── morse_encoder.py        # text -> Morse audio synthesis (the inverse pipeline)
│   └── pipeline.py             # wires the above into decode_wav_file()/decode_samples(), signal quality check
├── ui/                         # PyQt6 only - imports core, never the reverse
│   ├── main_window.py
│   ├── worker.py                # QThread for file decoding
│   ├── live_worker.py           # QThread for live/simulated mic decoding
│   ├── theme.py                 # Material 3 dark theme (colors + QSS)
│   └── widgets/waveform_widget.py
├── scripts/                    # standalone test/demo scripts, runnable headlessly
└── main.py
```

`core` never imports PyQt6; `ui` never does raw signal math. This means
the entire DSP pipeline can be run and tested from a plain script with no
GUI involved - see `scripts/`.

## Running the app

```bash
pip install -r requirements.txt
python main.py
```

On Linux, live microphone capture additionally needs the PortAudio system
library (`sudo apt-get install libportaudio2` on Debian/Ubuntu). Without
it, the app still runs fine for `.wav` file decoding - it just shows a
clear error if you click "Start Microphone".

## How decoding works, briefly

1. **Band-pass filter** the audio around the expected tone frequency
   (Butterworth, zero-phase for file decode; this is what "Tone Frequency"
   tunes).
2. **Envelope detection**: rectify and smooth with a moving average.
3. **Adaptive threshold + hysteresis binarization**: classify each sample
   as tone/silence, using two thresholds (not one) to resist chattering
   when the envelope hovers near the cutoff.
4. **Run-length encode** the binary signal into tone/silence durations,
   debounce short noise glitches, and estimate the timing unit (with
   outlier rejection so one bad glitch can't corrupt the whole message's
   calibration).
5. **Classify runs** into dots/dashes/gaps by duration relative to the
   estimated unit, then **decode** the resulting Morse string to text.

Live decoding runs this same pipeline repeatedly on a growing buffer, but
locks the threshold/timing calibration once there's enough audio to trust
it - otherwise, recalibrating from scratch on every poll would silently
reclassify already-decoded letters as new audio arrives.

## Tuning for your own recordings

- **Tone Frequency**: should match your actual Morse tone source. Use
  "Detect Frequency" rather than guessing.
- **Sensitivity**: how far above the noise floor a signal needs to be to
  count as "tone". Raise it if quiet background noise is triggering false
  detections; lower it if quiet real tone is being missed.

## Text \u2192 Morse encoder

The "Text \u2192 Morse (Encoder)" card at the bottom of the window is the
inverse of everything above: type a message, click Generate to see its
Morse notation, then Export WAV to save it as a real audio file (standard
timing - dot = 1 unit, dash = 3 units - with a short fade in/out on each
tone burst so it doesn't click, the same reason real CW transmitters use
a deliberate rise/fall time on their keying). This is a completely
separate module (`core/morse_encoder.py`) that only reads from
`morse_decoder.encode_text()` and `config.wpm_to_unit_seconds()` - it
doesn't share any code with the decode pipeline. Generated audio decodes
back to the original text correctly through the app's own decoder.

## Known limitation: recording quality matters more than tuning

No amount of threshold tuning can recover a recording where the actual
"silence" between beeps never drops much below the tone level. In
practice this showed up with **Bluetooth headset microphones**: they
switch into a call-oriented mode (mono, low sample rate, aggressive noise
suppression/echo cancellation) when the mic is active, which smears sharp
on/off transitions and can make quick repeated dots progressively
quieter. If the app's quality warning says separation is "weak" or
"poor", the fix is a better recording, not a different threshold:

- Best: capture the tone generator's audio output directly (e.g. via
  "Stereo Mix" or a virtual audio cable), avoiding a microphone entirely.
- Good: use a device's built-in microphone rather than a Bluetooth one.

## Fixed: timing calibration bug on dash-heavy messages

`estimate_unit_seconds()` (in `morse_state_machine.py`) previously
rejected outlier pulse durations by comparing them against the *overall
median* duration. This worked fine when dots outnumbered dashes (the
common case) but broke whenever dashes were the majority in a given
message (easy to hit - digits and letters like O/W/G/M/K/Y are
dash-heavy): the median itself shifted into dash-length territory, and
the entire real dot cluster got wrongly rejected as "outliers," breaking
timing calibration for the whole message. This affected many ordinary
messages, not just rare edge cases - e.g. "THANK YOU" and "PYTHON
PROGRAMMING" both failed under the old logic.

Fixed by checking the *relative size* of the cluster a split would
isolate, rather than comparing raw durations against a global statistic
that shifts with the dot/dash mix - a genuine dot cluster should be a
reasonable share of the pulses seen, not just one or two stray points.
This keeps the original protection against single noise glitches while
no longer breaking on ordinary dash-heavy text.

## Testing

Every phase has a standalone test script in `scripts/`, runnable without
a display:

```bash
python scripts/test_phase4_decoder.py path/to/file.wav [--window MS] [--ratio RATIO]
QT_QPA_PLATFORM=offscreen python scripts/test_phase7_live.py path/to/file.wav ["EXPECTED TEXT"]
QT_QPA_PLATFORM=offscreen python scripts/test_phase7_ui.py path/to/file.wav "EXPECTED TEXT" [screenshot.png]
```

`scripts/make_test_wav.py [output_path]` generates a synthetic "SOS"
test `.wav` file (a 700Hz tone in a fixed on/off pattern) - useful for
quick smoke tests without recording real audio.
