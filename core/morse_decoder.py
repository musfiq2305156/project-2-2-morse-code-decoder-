"""
Morse-to-text decoding.

Phase 4 scope: take the Morse-notation string produced by Phase 3
(e.g. ".... .. / --- -.-") and convert it into readable text
("HI OK"), using a standard International Morse Code lookup table.

This completes the headless core pipeline: a .wav file in, decoded
text out, with zero GUI dependency anywhere in the chain
(audio_io -> signal_processing -> morse_state_machine -> morse_decoder).
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# Standard International Morse Code lookup table.
# Keys are dot/dash patterns, values are the corresponding character.
# ---------------------------------------------------------------------------

MORSE_TO_CHAR = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",

    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",

    ".-.-.-": ".", "--..--": ",", "..--..": "?", ".----.": "'",
    "-.-.--": "!", "-..-.": "/", "-.--.": "(", "-.--.-": ")",
    ".-...": "&", "---...": ":", "-.-.-.": ";", "-...-": "=",
    ".-.-.": "+", "-....-": "-", "..--.-": "_", ".-..-.": '"',
    "...-..-": "$", ".--.-.": "@",
}

# Reverse table, useful for tooling/tests/future "encode text -> morse" features.
CHAR_TO_MORSE = {char: pattern for pattern, char in MORSE_TO_CHAR.items()}

UNKNOWN_CHAR_PLACEHOLDER = "\uFFFD"  # Unicode replacement character, "�"


def decode_letter(pattern: str) -> str:
    """
    Decode a single dot/dash pattern into its character.

    Returns UNKNOWN_CHAR_PLACEHOLDER for a pattern not in the table
    (e.g. from a timing misclassification) rather than raising, so one
    bad letter doesn't blow up decoding of an otherwise-good message.
    Returns an empty string for an empty pattern (defensive, e.g. from
    stray double-spaces in the Morse string).
    """
    if not pattern:
        return ""
    return MORSE_TO_CHAR.get(pattern, UNKNOWN_CHAR_PLACEHOLDER)


def decode_morse(morse: str) -> str:
    """
    Decode a full Morse-notation string into text.

    Expected format matches morse_state_machine.classify_runs() output:
        - letters are dot/dash groups
        - single spaces separate letters
        - '/' (surrounded by spaces) separates words

    e.g. ".... .. / --- -.-"  ->  "HI OK"

    Unrecognized letter patterns are decoded as the Unicode replacement
    character rather than dropped, so the position/count of characters
    in a garbled message is still visible to the user.

    Args:
        morse: Morse-notation string, as produced by
            morse_state_machine.signal_to_morse()["morse"].

    Returns:
        Decoded text. Empty string if `morse` is empty/whitespace-only.
    """
    morse = morse.strip()
    if not morse:
        return ""

    words = morse.split(" / ")
    decoded_words: List[str] = []

    for word in words:
        letters = word.split()  # split on any run of whitespace
        decoded_word = "".join(decode_letter(letter) for letter in letters)
        decoded_words.append(decoded_word)

    return " ".join(decoded_words)


def encode_text(text: str) -> str:
    """
    Convenience inverse of decode_morse(): text -> Morse notation.
    Not required by the detection pipeline, but useful for generating
    test signals and for a possible future "type text, hear Morse"
    feature. Characters with no Morse representation are skipped.
    """
    words = text.upper().split(" ")
    encoded_words = []
    for word in words:
        letters = [CHAR_TO_MORSE[ch] for ch in word if ch in CHAR_TO_MORSE]
        encoded_words.append(" ".join(letters))
    return " / ".join(encoded_words)
