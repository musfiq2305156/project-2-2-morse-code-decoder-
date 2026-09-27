# Signals and Linear Systems - Project Theory & Implementation Guide

This document provides a detailed explanation of how fundamental **Signals and Linear Systems** course concepts are applied and implemented in this Morse Code Decoder/Encoder project.

---

## 1. Sampling Theorem & Discrete-Time Signal Processing

### Theory
Audio in the real world is a continuous-time signal $x(t)$. For digital computer processing, it is sampled according to the **Nyquist-Shannon Sampling Theorem** at a standard sampling rate $F_s = 44.1 \text{ kHz}$, converting it into a discrete-time signal:
$$x[n] = x(n T_s), \quad T_s = \frac{1}{F_s}$$

### Implementation
- **Files:** [`core/config.py`](core/config.py) and [`core/audio_io.py`](core/audio_io.py)
- `AudioBuffer` encapsulates discrete 1-D NumPy arrays normalized to floating-point values in the range $[-1.0, 1.0]$.
- `MicrophoneStream` and `load_wav_file()` handle continuous-to-discrete conversion for live audio input and WAV files respectively.

### Purpose
Converts continuous analog acoustic waves into discrete numerical arrays so DSP algorithms can process them mathematically without aliasing or information loss.

---

## 2. LTI Systems & Band-pass Filtering (Frequency Filtering)

### Theory
A **Linear Time-Invariant (LTI) System** filters signals by allowing desired frequency bands to pass while attenuating unwanted frequency components and noise. A 4th-order Butterworth bandpass filter centered around the target Morse tone frequency (e.g. $700 \text{ Hz}$) is used.

### Implementation
- **Batch Processing:** [`core/signal_processing.py`](core/signal_processing.py) using `scipy.signal.butter` and `filtfilt`. Zero-phase filtering (`filtfilt`) avoids phase distortion and time-shifting of Morse pulse edges.
- **Real-Time Live Mode:** [`core/live_decoder.py`](core/live_decoder.py) using Causal Second-Order Sections filtering (`sosfilt`).

### Purpose
Noise reduction. Isolates the specific Morse tone frequency band (~700 Hz) and rejects background room hum, fan noise, and speech frequencies.

---

## 3. Discrete Convolution & Moving Average Filtering (Envelope Detection)

### Theory
The **Discrete-Time Convolution** equation between signal $x[n]$ and impulse response $h[n]$:
$$y[n] = (x * h)[n] = \sum_{k=-\infty}^{\infty} x[k] h[n-k]$$

Full-wave rectifying the filtered waveform ($|x[n]|$) and convolving it with a rectangular moving-average kernel $h[n]$ functions as a low-pass filter to extract the smooth amplitude envelope $E[n]$.

### Implementation
- **File:** [`core/signal_processing.py`](core/signal_processing.py) in `compute_envelope()`
- Code: `np.convolve(rectified, kernel, mode="same")`

### Purpose
Converts high-frequency sinusoidal oscillations (~700 Hz) into a smooth, non-negative amplitude envelope representing tone power over time.

---

## 4. Frequency Domain Analysis & Fast Fourier Transform (FFT)

### Theory
Converting a time-domain signal $x[n]$ into its frequency spectrum $X[k]$ using the **Discrete Fourier Transform (DFT)** via the Fast Fourier Transform (FFT) algorithm:
$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi}{N} k n}$$

### Implementation
- **File:** [`core/signal_processing.py`](core/signal_processing.py) in `detect_dominant_frequency()`
- Code: Energy masking + Hanning windowing + `np.fft.rfft(windowed)`

### Purpose
**Auto-Tuning:** Automatically detects the exact frequency peak (Hz) of the Morse tone in an input recording so the band-pass filter centers accurately without manual guessing.

---

## 5. Non-linear Binarization & Hysteresis (Schmitt Trigger Principle)

### Theory
Converting the continuous amplitude envelope $E[n]$ into a discrete binary pulse signal $B[n] \in \{0, 1\}$. Uses two decision thresholds (**Hysteresis**) to prevent chatter/flickering caused by noise fluctuations near threshold boundaries.

### Implementation
- **File:** [`core/signal_processing.py`](core/signal_processing.py) in `adaptive_threshold()` and `binarize()`
- Statistical noise-floor sigma estimation combined with connected component labeling (`scipy.ndimage.label`).

### Purpose
Robustly maps the analog amplitude envelope into binary square waves representing Tone ON ($1$) vs Silence OFF ($0$).

---

## 6. Discrete-Time State Tracking & Pulse Timing Classification

### Theory
Discrete-time run-length encoding (RLE) and timing interval analysis. Measures duration $T_i$ of continuous binary runs to classify pulses into standard International Morse Code timing ratios:
- Dot ($\cdot$) = $1 \text{ Unit}$
- Dash ($-$) = $3 \text{ Units}$
- Intra-character Gap = $1 \text{ Unit}$
- Inter-character Gap = $3 \text{ Units}$
- Word Gap = $7 \text{ Units}$

### Implementation
- **Files:** [`core/morse_state_machine.py`](core/morse_state_machine.py) and [`core/live_decoder.py`](core/live_decoder.py)
- Features glitch cleaning (`MIN_PULSE_MS`), outlier-resistant WPM unit calibration, and state machine decoding.

### Purpose
Decodes binary pulse timing into dots, dashes, and spaces, and converts them to readable English text using lookup tables ([`core/morse_decoder.py`](core/morse_decoder.py)).

---

## 7. Sinusoidal Signal Synthesis & Envelope Shaping (Audio Generation)

### Theory
Synthesizing a discrete sinusoidal carrier wave $s[n] = A \sin(2\pi f t)$ modulated by Morse pulse timing, applying cosine ramp tapering to prevent high-frequency spectral leakage and clicking.

### Implementation
- **File:** [`core/morse_encoder.py`](core/morse_encoder.py)

### Purpose
**Inverse Pipeline:** Turns input text into Morse notation and generates valid `.wav` audio files.

---

## Summary Mapping Table

| Signals & Systems Topic | Core Implementation File | Practical Purpose |
| :--- | :--- | :--- |
| **Sampling & Discrete Signal** | [`core/audio_io.py`](core/audio_io.py) | Continuous audio wave to discrete sample arrays |
| **Butterworth Band-pass Filter** | [`core/signal_processing.py`](core/signal_processing.py) | Isolate tone frequency & reject noise |
| **Convolution (Moving Average)** | [`core/signal_processing.py`](core/signal_processing.py) | Extract smooth amplitude envelope |
| **Fast Fourier Transform (FFT)** | [`core/signal_processing.py`](core/signal_processing.py) | Auto-detect dominant tone frequency |
| **Hysteresis Binarization** | [`core/signal_processing.py`](core/signal_processing.py) | Convert envelope to clean Tone ON/OFF binary pulses |
| **Discrete State Machine** | [`core/morse_state_machine.py`](core/morse_state_machine.py) | Classify pulse timing into Morse dots, dashes & spaces |
| **Sinusoidal Signal Synthesis** | [`core/morse_encoder.py`](core/morse_encoder.py) | Generate `.wav` Morse audio files from text (Inverse Pipeline) |
