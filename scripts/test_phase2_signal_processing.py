"""
Phase 2 smoke test: loads a WAV, runs it through the amplitude
analysis pipeline, and plots three stacked views so you can visually
confirm the envelope tracks the tone and the threshold cleanly
separates tone from silence.

Usage:
    python scripts/make_test_wav.py test_signal.wav
    python scripts/test_phase2_signal_processing.py test_signal.wav
    python scripts/test_phase2_signal_processing.py test_signal.wav --window 15 --ratio 0.25
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from core.audio_io import load_wav_file
from core.signal_processing import analyze


def main():
    parser = argparse.ArgumentParser(description="Phase 2: amplitude envelope + thresholding test")
    parser.add_argument("wav_path", help="Path to a .wav file")
    parser.add_argument("--window", type=float, default=10.0, help="Moving-average window in ms (default: 10)")
    parser.add_argument("--ratio", type=float, default=0.3, help="Threshold as fraction of peak envelope (default: 0.3)")
    parser.add_argument("--out", default=None, help="If set, save the plot to this path instead of showing it")
    args = parser.parse_args()

    buffer = load_wav_file(args.wav_path)
    result = analyze(buffer, window_ms=args.window, threshold_ratio=args.ratio)

    t = np.arange(len(buffer.samples)) / buffer.sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(t, buffer.samples, linewidth=0.5, color="#1E5FFF")
    axes[0].set_title("Raw waveform")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t, result["envelope"], linewidth=1.0, color="#1E5FFF")
    axes[1].axhline(result["threshold"], color="red", linestyle="--", linewidth=1.0,
                     label=f"threshold = {result['threshold']:.4f}")
    axes[1].set_title(f"Amplitude envelope (moving avg, window={args.window} ms)")
    axes[1].set_ylabel("Envelope")
    axes[1].legend(loc="upper right")

    axes[2].plot(t, result["binary"].astype(int), linewidth=1.2, color="#1E5FFF", drawstyle="steps-post")
    axes[2].set_title("Binary tone/silence signal")
    axes[2].set_ylabel("Tone (1/0)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(-0.2, 1.2)

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved plot to '{args.out}'")
    else:
        plt.show()

    # Also print a quick numeric summary, useful when running headless.
    n_tone_samples = int(np.sum(result["binary"]))
    print(f"threshold        : {result['threshold']:.5f}")
    print(f"tone samples     : {n_tone_samples} / {len(result['binary'])} "
          f"({100 * n_tone_samples / len(result['binary']):.1f}%)")


if __name__ == "__main__":
    main()
