"""APF V1 -- Speech-to-Text wrapper (M6).

V1 uses browser-based Web Speech API (no backend STT needed).
V2+ can integrate Whisper for Telugu/code-mixed support.

This module provides a placeholder for future Whisper integration.
"""


def transcribe_audio(audio_bytes: bytes, language: str = "en") -> str:
    """Transcribe audio bytes to text.

    V1: Not used -- browser handles STT via Web Speech API.
    V2: Will integrate Whisper for Telugu/code-mixed support.

    Args:
        audio_bytes: Raw audio file bytes (wav/mp3)
        language: Language code (en, te, hi, etc.)

    Returns:
        Transcribed text
    """
    raise NotImplementedError(
        "V1 uses browser-based Web Speech API. "
        "Backend STT (Whisper) will be implemented in V2."
    )


def is_whisper_available() -> bool:
    """Check if Whisper is installed and available."""
    try:
        import whisper
        return True
    except ImportError:
        return False


def get_whisper_model_size() -> str:
    """Return recommended Whisper model size based on available resources."""
    try:
        import torch
        if torch.cuda.is_available():
            return "medium"
    except ImportError:
        pass
    return "base"
