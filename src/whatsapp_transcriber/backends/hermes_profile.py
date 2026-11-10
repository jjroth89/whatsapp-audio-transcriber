import os
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from .base import BaseTranscriber

class HermesProfileTranscriber(BaseTranscriber):
    """STT backend invoking a Hermes profile STT service or OpenAI-compatible endpoint."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.endpoint_url = (
            endpoint_url
            or os.environ.get("HERMES_STT_ENDPOINT")
            or os.environ.get("OPENAI_AUDIO_ENDPOINT")
            or "http://localhost:8080/v1/audio/transcriptions"
        )
        self.api_key = api_key or os.environ.get("HERMES_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.model_name = model_name or os.environ.get("HERMES_STT_MODEL") or "whisper-1"

    def transcribe(
        self,
        audio_path: Path,
        prompt: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": self.model_name,
            "response_format": "verbose_json"
        }
        if prompt:
            data["prompt"] = prompt
        if language:
            data["language"] = language

        try:
            with audio_path.open("rb") as audio_file:
                files = {"file": (audio_path.name, audio_file)}
                response = requests.post(
                    self.endpoint_url,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=120
                )
                response.raise_for_status()
                res_data = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to transcribe via Hermes STT profile at {self.endpoint_url}: {exc}"
            ) from exc

        text = res_data.get("text", "").strip()
        duration = round(float(res_data.get("duration", 0.0)), 2)
        raw_segs = res_data.get("segments", [])

        normalized_segs = []
        for s in raw_segs:
            normalized_segs.append({
                "start": round(float(s.get("start", 0.0)), 3),
                "end": round(float(s.get("end", 0.0)), 3),
                "text": s.get("text", "").strip(),
                "avg_logprob": round(float(s.get("avg_logprob", 0.0)), 3) if "avg_logprob" in s else None,
                "no_speech_prob": round(float(s.get("no_speech_prob", 0.0)), 3) if "no_speech_prob" in s else None,
                "words": s.get("words", [])
            })

        alerts = []
        if not text:
            alerts.append("inaudible_or_silent")

        return {
            "text": text,
            "duration": duration,
            "language": res_data.get("language", language or "pt"),
            "confidence": 1.0,
            "segments": normalized_segs,
            "alerts": alerts,
            "engine": f"hermes-stt/{self.model_name} ({self.endpoint_url})"
        }
