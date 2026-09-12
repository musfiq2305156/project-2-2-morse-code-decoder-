"""
Main application window.

Phase 5 added: "Load WAV File" button, background decoding thread, and a
decoded-text/Morse-notation result card.

Phase 6 added: an embedded waveform + envelope/threshold view above the
result card.

Phase 7 scope: live audio decoding. A "Start Microphone" toggle captures
from the system's default input device and continuously re-decodes a
rolling buffer, updating the waveform and decoded text as audio arrives.
A second "Simulate from File" button drives the exact same live pipeline
from a pre-recorded .wav instead of a physical mic - useful as a demo
mode and for testing on machines with no microphone or no PortAudio
install (this project's own dev environment had neither, which is how
that fallback path ended up getting used and verified).
"""

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QPushButton,
    QFileDialog,
    QSpinBox,
    QSlider,
)
from PyQt6.QtCore import Qt

from ui.theme import COLORS
from ui.worker import DecodeWorker
from ui.live_worker import LiveDecodeWorker
from ui.widgets.waveform_widget import WaveformWidget
from core.audio_io import MicrophoneStream, SimulatedMicrophoneStream
from core.config import DEFAULT_TONE_FREQ_HZ, DEFAULT_THRESHOLD_RATIO
from core.pipeline import detect_tone_frequency_from_wav, detect_tone_frequency_from_samples


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Morse Code Detector")
        self.resize(960, 700)
        self._worker = None       # active DecodeWorker (file decode), if any
        self._live_worker = None  # active LiveDecodeWorker (mic/simulated), if any
        self._last_wav_path = None  # most recently loaded/simulated .wav, for "Detect Frequency"
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Morse Code Detector")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")

        subtitle = QLabel("Load a .wav file, or decode live from the microphone.")
        subtitle.setProperty("role", "subtitle")

        # --- file loading row -----------------------------------------
        load_row = QHBoxLayout()
        self.load_button = QPushButton("Load WAV File")
        self.load_button.clicked.connect(self._on_load_clicked)

        self.file_label = QLabel("No file loaded")
        self.file_label.setProperty("role", "subtitle")

        load_row.addWidget(self.load_button)
        load_row.addWidget(self.file_label)
        load_row.addStretch()

        # --- live capture row --------------------------------------------
        live_row = QHBoxLayout()
        self.mic_button = QPushButton("Start Microphone")
        self.mic_button.clicked.connect(self._on_mic_toggle_clicked)

        self.simulate_button = QPushButton("Simulate from File")
        self.simulate_button.clicked.connect(self._on_simulate_clicked)

        live_row.addWidget(self.mic_button)
        live_row.addWidget(self.simulate_button)
        live_row.addStretch()

        # --- detection tuning row -----------------------------------------
        tuning_row = QHBoxLayout()

        tone_freq_label = QLabel("Tone Frequency (Hz):")
        self.tone_freq_spinbox = QSpinBox()
        self.tone_freq_spinbox.setRange(200, 2000)
        self.tone_freq_spinbox.setSingleStep(10)
        self.tone_freq_spinbox.setValue(int(DEFAULT_TONE_FREQ_HZ))
        self.tone_freq_spinbox.setToolTip(
            "Should match the pitch of your actual Morse tone source "
            "(keyer, oscillator, practice app). If it's off by more than "
            "~100 Hz from the real tone, detection gets worse, not better."
        )

        self.detect_freq_button = QPushButton("Detect Frequency")
        self.detect_freq_button.setToolTip(
            "Analyzes the loaded file (or, during a live session, the "
            "last few seconds captured so far) to find the actual tone "
            "frequency, instead of guessing."
        )
        self.detect_freq_button.clicked.connect(self._on_detect_frequency_clicked)

        sensitivity_label = QLabel("Sensitivity:")
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(10, 60)  # maps to threshold_ratio 0.10-0.60
        self.sensitivity_slider.setValue(int(DEFAULT_THRESHOLD_RATIO * 100))
        self.sensitivity_slider.setFixedWidth(140)
        self.sensitivity_value_label = QLabel(f"{DEFAULT_THRESHOLD_RATIO:.2f}")
        self.sensitivity_value_label.setProperty("role", "subtitle")
        self.sensitivity_slider.valueChanged.connect(
            lambda v: self.sensitivity_value_label.setText(f"{v / 100:.2f}")
        )
        self.sensitivity_slider.setToolTip(
            "Lower = more sensitive to quiet signals, but more prone to "
            "false triggers on background noise. Higher = requires a "
            "stronger, cleaner tone."
        )

        tuning_row.addWidget(tone_freq_label)
        tuning_row.addWidget(self.tone_freq_spinbox)
        tuning_row.addWidget(self.detect_freq_button)
        tuning_row.addSpacing(24)
        tuning_row.addWidget(sensitivity_label)
        tuning_row.addWidget(self.sensitivity_slider)
        tuning_row.addWidget(self.sensitivity_value_label)
        tuning_row.addStretch()

        # --- status line (idle / decoding.../ listening.../ error) -----
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {COLORS['primary']}; font-weight: 600;")

        # --- waveform card ------------------------------------------------
        waveform_card = QFrame()
        waveform_card.setProperty("role", "card")
        waveform_card.setMinimumHeight(260)
        waveform_layout = QVBoxLayout()
        waveform_layout.setContentsMargins(16, 16, 16, 16)
        self.waveform_widget = WaveformWidget()
        waveform_layout.addWidget(self.waveform_widget)
        waveform_card.setLayout(waveform_layout)

        # --- decoded output card ----------------------------------------
        result_card = QFrame()
        result_card.setProperty("role", "card")
        result_card.setMinimumHeight(280)
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(16)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        decoded_heading = QLabel("Decoded Text")
        decoded_heading.setStyleSheet("font-size: 13px; font-weight: 600; "
                                       f"color: {COLORS['on_surface_variant']};")

        self.decoded_text_label = QLabel("\u2014")
        self.decoded_text_label.setWordWrap(True)
        self.decoded_text_label.setStyleSheet("font-size: 32px; font-weight: 700;")

        morse_heading = QLabel("Morse Notation")
        morse_heading.setStyleSheet("font-size: 13px; font-weight: 600; "
                                     f"color: {COLORS['on_surface_variant']};")

        self.morse_label = QLabel("\u2014")
        self.morse_label.setWordWrap(True)
        self.morse_label.setStyleSheet(f"font-size: 16px; font-family: monospace; "
                                        f"color: {COLORS['on_surface_variant']};")

        self.meta_label = QLabel("")
        self.meta_label.setStyleSheet(f"font-size: 12px; color: {COLORS['on_surface_variant']};")

        card_layout.addWidget(decoded_heading)
        card_layout.addWidget(self.decoded_text_label)
        card_layout.addWidget(morse_heading)
        card_layout.addWidget(self.morse_label)
        card_layout.addStretch()
        card_layout.addWidget(self.meta_label)
        result_card.setLayout(card_layout)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(load_row)
        layout.addLayout(live_row)
        layout.addLayout(tuning_row)
        layout.addWidget(self.status_label)
        layout.addWidget(waveform_card)
        layout.addWidget(result_card)

        central.setLayout(layout)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str, is_error: bool = False) -> None:
        color = COLORS["error"] if is_error else COLORS["primary"]
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _clear_result(self) -> None:
        self.decoded_text_label.setText("\u2014")
        self.morse_label.setText("\u2014")
        self.meta_label.setText("")
        self.waveform_widget.clear()

    def _display_result(self, result: dict) -> None:
        """Shared by file decoding and live decoding - both produce the
        same result dict shape (core.pipeline), so both render identically."""
        text = result["text"] or "(no message detected)"
        self.decoded_text_label.setText(text)
        self.morse_label.setText(result["morse"] or "(none)")
        self.meta_label.setText(
            f"duration: {result['duration']:.2f}s   "
            f"sample rate: {result['sample_rate']} Hz   "
            f"estimated unit: {result['unit_seconds'] * 1000:.1f} ms"
        )
        self.waveform_widget.plot(
            samples=result["samples"],
            sample_rate=result["sample_rate"],
            envelope=result["envelope"],
            threshold=result["threshold"],
        )

    def _set_controls_enabled(self, *, load: bool, mic: bool, simulate: bool) -> None:
        self.load_button.setEnabled(load)
        self.mic_button.setEnabled(mic)
        self.simulate_button.setEnabled(simulate)

    def _current_tuning(self) -> dict:
        """Read the current tone-frequency/sensitivity control values, to
        pass into a DecodeWorker or LiveDecodeWorker when starting a session."""
        return {
            "tone_freq_hz": float(self.tone_freq_spinbox.value()),
            "threshold_ratio": self.sensitivity_slider.value() / 100.0,
        }

    def _on_detect_frequency_clicked(self) -> None:
        """
        Analyze whatever audio is currently available - the live
        session's buffer if one is running, otherwise the last loaded/
        simulated .wav file - to find the actual tone frequency, and
        fill the spinbox with it instead of leaving the person to guess.
        """
        try:
            if self._live_worker is not None:
                samples, sample_rate = self._live_worker.get_buffer_snapshot()
                if samples.size < sample_rate * 0.5:
                    self._set_status("Detect Frequency: not enough live audio captured yet", is_error=True)
                    return
                result = detect_tone_frequency_from_samples(samples, sample_rate)
            elif self._last_wav_path:
                result = detect_tone_frequency_from_wav(self._last_wav_path)
            else:
                self._set_status("Detect Frequency: load a file or start a session first", is_error=True)
                return
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Detect Frequency failed: {exc}", is_error=True)
            return

        detected_hz = result["frequency_hz"]
        confidence = result["confidence"]
        self.tone_freq_spinbox.setValue(int(round(detected_hz)))

        if confidence < 3.0:
            # A flat/ambiguous spectrum (no clear peak) usually means
            # there wasn't much real tone in the analyzed audio yet -
            # e.g. it was mostly silence, or capture just started.
            self._set_status(
                f"Detected {detected_hz:.0f} Hz, but with low confidence "
                f"({confidence:.1f}x) - try again with more tone captured",
                is_error=True,
            )
        else:
            self._set_status(f"Detected tone frequency: {detected_hz:.0f} Hz (confidence {confidence:.0f}x)")

    # ------------------------------------------------------------------
    # File decode (Phase 5/6)
    # ------------------------------------------------------------------

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a WAV file", "", "WAV files (*.wav)"
        )
        if not path:
            return  # user cancelled the dialog

        self.file_label.setText(path)
        self._last_wav_path = path
        self._start_decode(path)

    def _start_decode(self, path: str) -> None:
        self._set_controls_enabled(load=False, mic=False, simulate=False)
        self._set_status("Decoding\u2026")
        self._clear_result()

        self._worker = DecodeWorker(path, **self._current_tuning())
        self._worker.finished.connect(self._on_decode_finished)
        self._worker.failed.connect(self._on_decode_failed)
        self._worker.start()

    def _on_decode_finished(self, result: dict) -> None:
        self._set_controls_enabled(load=True, mic=True, simulate=True)
        self._set_status("Done")
        self._display_result(result)

    def _on_decode_failed(self, message: str) -> None:
        self._set_controls_enabled(load=True, mic=True, simulate=True)
        self._set_status("Error", is_error=True)
        self._clear_result()
        self.meta_label.setText(f"Could not decode file: {message}")

    # ------------------------------------------------------------------
    # Live decode (Phase 7)
    # ------------------------------------------------------------------

    def _on_mic_toggle_clicked(self) -> None:
        if self._live_worker is not None:
            self._stop_live()
            return
        self._start_live(MicrophoneStream(), listening_label="Listening\u2026 (microphone)")

    def _on_simulate_clicked(self) -> None:
        if self._live_worker is not None:
            self._stop_live()
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select a WAV file to simulate as live input", "", "WAV files (*.wav)"
        )
        if not path:
            return  # user cancelled the dialog

        self._last_wav_path = path
        self._start_live(
            SimulatedMicrophoneStream(path),
            listening_label=f"Listening\u2026 (simulating {path})",
        )

    def _start_live(self, stream, listening_label: str) -> None:
        self._set_controls_enabled(load=False, mic=True, simulate=True)
        # Whichever button started the session becomes "Stop ..."; the
        # other live-input button is disabled so the two can't collide.
        if isinstance(stream, MicrophoneStream):
            self.mic_button.setText("Stop Microphone")
            self.simulate_button.setEnabled(False)
        else:
            self.simulate_button.setText("Stop Simulation")
            self.mic_button.setEnabled(False)

        self._set_status(listening_label)
        self._clear_result()

        self._live_worker = LiveDecodeWorker(stream, **self._current_tuning())
        self._live_worker.updated.connect(self._on_live_updated)
        self._live_worker.failed.connect(self._on_live_failed)
        self._live_worker.start()

    def _stop_live(self) -> None:
        if self._live_worker is not None:
            self._live_worker.updated.disconnect(self._on_live_updated)
            self._live_worker.failed.disconnect(self._on_live_failed)
            self._live_worker.stop()
            self._live_worker = None

        self.mic_button.setText("Start Microphone")
        self.simulate_button.setText("Simulate from File")
        self._set_controls_enabled(load=True, mic=True, simulate=True)
        self._set_status("Stopped")

    def _on_live_updated(self, result: dict) -> None:
        self._set_status(self.status_label.text())  # keep existing "Listening..." text/color
        self._display_result(result)

    def _on_live_failed(self, message: str) -> None:
        self._live_worker = None
        self.mic_button.setText("Start Microphone")
        self.simulate_button.setText("Simulate from File")
        self._set_controls_enabled(load=True, mic=True, simulate=True)
        self._set_status("Error", is_error=True)
        self._clear_result()
        self.meta_label.setText(f"Live input error: {message}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        # Make sure a live session doesn't keep the audio stream (and its
        # background thread) running after the window closes.
        if self._live_worker is not None:
            self._live_worker.stop()
            self._live_worker = None
        super().closeEvent(event)
