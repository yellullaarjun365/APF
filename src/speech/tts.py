"""APF V1 -- Text-to-Speech wrapper (M6).

V1 uses browser-based Web Speech API (no backend TTS needed).
V2+ can integrate pyttsx3 or cloud TTS for better quality.

This module provides a placeholder for future TTS integration.
"""


def synthesize_speech(text: str, language: str = "en") -> bytes:
    """Synthesize text to audio bytes.

    V1: Not used -- browser handles TTS via Web Speech API.
    V2: Will integrate pyttsx3 or cloud TTS.

    Args:
        text: Text to synthesize
        language: Language code

    Returns:
        Audio file bytes (wav format)
    """
    raise NotImplementedError(
        "V1 uses browser-based Web Speech API. "
        "Backend TTS will be implemented in V2."
    )


def is_pyttsx3_available() -> bool:
    """Check if pyttsx3 is installed."""
    try:
        import pyttsx3
        return True
    except ImportError:
        return False
