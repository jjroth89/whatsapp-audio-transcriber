from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

class BaseTranscriber(ABC):
    """Abstract interface for STT transcription backends."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        prompt: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe a single audio file.

        Returns a dictionary containing:
            - text: str
            - duration: float
            - language: str
            - confidence: float
            - segments: list[dict]
            - alerts: list[str]
            - engine: str
        """
        pass
