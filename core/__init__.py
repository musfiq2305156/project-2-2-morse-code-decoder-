"""
Core signal-processing package.

IMPORTANT PROJECT RULE:
Nothing in this package may import PyQt6 (or any UI toolkit). This package
must remain testable and runnable from a plain Python script / CLI, with
no GUI dependency whatsoever. The `ui` package depends on `core`, never
the other way around.
"""
