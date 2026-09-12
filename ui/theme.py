"""
Material 3 inspired theme: deep blue accent, matte black background,
white text - matching the Pixel "dark, expressive Material You" look.

Kept as plain color tokens + a QSS string so later widgets (Phase 5+)
just import COLORS or apply STYLESHEET, without redefining palette values
in scattered places.
"""

# ---------------------------------------------------------------------------
# Color tokens (Material 3 naming convention, dark scheme)
# ---------------------------------------------------------------------------

COLORS = {
    "background": "#0A0A0C",        # matte black
    "surface": "#141417",           # slightly raised panels
    "surface_variant": "#1D1D21",   # cards, input fields
    "outline": "#2E2E33",           # subtle borders/dividers
    "primary": "#1E5FFF",           # deep blue accent
    "primary_container": "#152A5C", # muted blue for containers/hover
    "on_primary": "#FFFFFF",
    "on_background": "#FFFFFF",     # primary text
    "on_surface_variant": "#B8B8C0",# secondary/muted text
    "error": "#FF5449",
}


# ---------------------------------------------------------------------------
# QSS stylesheet
# ---------------------------------------------------------------------------

STYLESHEET = f"""
QWidget {{
    background-color: {COLORS['background']};
    color: {COLORS['on_background']};
    font-family: "Roboto", "Segoe UI", sans-serif;
    font-size: 14px;
}}

QMainWindow {{
    background-color: {COLORS['background']};
}}

QLabel {{
    color: {COLORS['on_background']};
    background-color: transparent;
}}

QLabel[role="subtitle"] {{
    color: {COLORS['on_surface_variant']};
    font-size: 13px;
}}

QPushButton {{
    background-color: {COLORS['primary']};
    color: {COLORS['on_primary']};
    border: none;
    border-radius: 20px;
    padding: 10px 24px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: #3A76FF;
}}

QPushButton:pressed {{
    background-color: {COLORS['primary_container']};
}}

QPushButton:disabled {{
    background-color: {COLORS['surface_variant']};
    color: {COLORS['on_surface_variant']};
}}

QFrame[role="card"] {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['outline']};
    border-radius: 16px;
}}

QFrame[role="card"] QWidget {{
    background-color: transparent;
}}

QSlider::groove:horizontal {{
    background: {COLORS['surface_variant']};
    height: 4px;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {COLORS['primary']};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}

QScrollBar:vertical {{
    background: {COLORS['background']};
    width: 8px;
}}

QScrollBar::handle:vertical {{
    background: {COLORS['outline']};
    border-radius: 4px;
}}
"""
