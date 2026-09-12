"""
Waveform visualization widget.

Phase 6 scope: embed a matplotlib figure inside the PyQt window showing
the raw waveform, the Phase 2 amplitude envelope, and the threshold line
used to separate tone from silence - giving a visual confirmation of
what the detector is actually reacting to, styled to match the app's
Material 3 dark theme.

This widget is purely a "given data, draw it" component: it takes plain
numpy arrays (as already returned by core.pipeline.decode_wav_file) and
never touches core logic itself, keeping the ui/core boundary intact.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from ui.theme import COLORS

# Cap the number of points actually drawn, independent of how many
# samples are in the buffer. A multi-second recording at 44.1kHz is
# hundreds of thousands of samples - matplotlib doesn't need to render
# every single one for a static overview plot, and this keeps redraws
# fast even on longer files (increasingly relevant once Phase 7 streams
# from a live microphone).
MAX_PLOT_POINTS = 6000


class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure = Figure(figsize=(5, 3), facecolor=COLORS["surface"])
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.ax_waveform = self.figure.add_subplot(211)
        self.ax_envelope = self.figure.add_subplot(212, sharex=self.ax_waveform)
        self.figure.subplots_adjust(hspace=0.45, left=0.08, right=0.97, top=0.93, bottom=0.15)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self._style_axes()
        self.clear()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

    def _style_axes(self) -> None:
        for ax in (self.ax_waveform, self.ax_envelope):
            ax.set_facecolor(COLORS["surface"])
            ax.tick_params(colors=COLORS["on_surface_variant"], labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(COLORS["outline"])
            ax.xaxis.label.set_color(COLORS["on_surface_variant"])
            ax.yaxis.label.set_color(COLORS["on_surface_variant"])
            ax.title.set_color(COLORS["on_surface_variant"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset to an empty placeholder state (e.g. before any file is loaded)."""
        self.ax_waveform.clear()
        self.ax_envelope.clear()
        self._style_axes()
        self.ax_waveform.set_title("Waveform", fontsize=9, loc="left")
        self.ax_envelope.set_title("Envelope + threshold", fontsize=9, loc="left")
        self.ax_envelope.set_xlabel("Time (s)", fontsize=8)
        self.canvas.draw_idle()

    def plot(self, samples: np.ndarray, sample_rate: int,
              envelope: np.ndarray, threshold: float) -> None:
        """
        Draw the waveform and envelope/threshold for a decoded file.

        Args:
            samples: normalized raw waveform, as returned by
                core.pipeline.decode_wav_file()["samples"].
            sample_rate: samples per second.
            envelope: smoothed amplitude envelope, same length as samples.
            threshold: scalar tone/silence cutoff to draw as a reference line.
        """
        self.ax_waveform.clear()
        self.ax_envelope.clear()
        self._style_axes()

        t = np.arange(len(samples)) / sample_rate
        t_plot, samples_plot = _downsample_for_plot(t, samples)
        _, envelope_plot = _downsample_for_plot(t, envelope)

        self.ax_waveform.plot(t_plot, samples_plot, color=COLORS["primary"], linewidth=0.6)
        self.ax_waveform.set_title("Waveform", fontsize=9, loc="left")
        self.ax_waveform.set_ylim(-1.05, 1.05)

        self.ax_envelope.plot(t_plot, envelope_plot, color=COLORS["primary"], linewidth=1.0)
        self.ax_envelope.axhline(threshold, color="#FF6B5E", linestyle="--", linewidth=1.0,
                                  label=f"threshold = {threshold:.3f}")
        self.ax_envelope.legend(loc="upper right", fontsize=7, facecolor=COLORS["surface_variant"],
                                 edgecolor=COLORS["outline"], labelcolor=COLORS["on_background"])
        self.ax_envelope.set_title("Envelope + threshold", fontsize=9, loc="left")
        self.ax_envelope.set_xlabel("Time (s)", fontsize=8)

        self.canvas.draw_idle()


def _downsample_for_plot(t: np.ndarray, y: np.ndarray, max_points: int = MAX_PLOT_POINTS):
    """
    Evenly stride down to at most `max_points` samples for fast, still
    visually faithful plotting. Uses plain slicing (not averaging/min-max
    binning) - fine for an overview plot; a future phase could switch to
    min/max envelope binning if individual short pulses ever get
    stepped over at this resolution.
    """
    n = len(y)
    if n <= max_points:
        return t, y
    stride = max(1, n // max_points)
    return t[::stride], y[::stride]
