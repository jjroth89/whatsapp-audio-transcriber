import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from .base import BaseTranscriber

def _configure_windows_cuda_dlls():
    """Ensure CUDA and cuDNN DLLs installed via pip packages are discoverable."""
    if sys.platform != "win32":
        return
    site_packages = Path(sys.executable).parent.parent / "Lib" / "site-packages" / "nvidia"
    if site_packages.exists():
        for subfolder in ["cublas", "cudnn", "cuda_nvrtc"]:
            bin_path = site_packages / subfolder / "bin"
            if bin_path.exists():
                str_bin = str(bin_path)
                os.environ["PATH"] = str_bin + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(str_bin)
                    except Exception:
                        pass

_configure_windows_cuda_dlls()

class LocalWhisperTranscriber(BaseTranscriber):
    """Local faster-whisper inference running on NVIDIA CUDA or CPU."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 8,
        beam_size: int = 5,
        cache_dir: Optional[Path] = None
    ):
        try:
            import ctranslate2
            from faster_whisper import WhisperModel, BatchedInferencePipeline
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper and ctranslate2 are required for LocalWhisperTranscriber. "
                "Install them with: pip install faster-whisper"
            ) from e

        if device == "cuda":
            cuda_count = ctranslate2.get_cuda_device_count()
            if cuda_count < 1:
                raise RuntimeError(
                    "CUDA device requested but no CUDA-capable GPU detected by CTranslate2. "
                    "Ensure NVIDIA drivers and CUDA libraries are installed."
                )

        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.beam_size = beam_size

        download_root = str(cache_dir) if cache_dir else None
        self._raw_model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=max(1, min(12, (os.cpu_count() or 8) // 2)),
            num_workers=2,
            download_root=download_root
        )
        self._pipeline = BatchedInferencePipeline(model=self._raw_model)

    def transcribe(
        self,
        audio_path: Path,
        prompt: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        target_lang = language or "pt"

        def _run(use_vad: bool):
            kwargs = {
                "language": target_lang,
                "beam_size": self.beam_size,
                "vad_filter": use_vad,
                "word_timestamps": True,
                "condition_on_previous_text": False,
                "initial_prompt": prompt,
                "temperature": 0.0,
                "batch_size": self.batch_size
            }
            seg_iter, info = self._pipeline.transcribe(str(audio_path), **kwargs)
            collected_segs = []
            for s in seg_iter:
                words = []
                for w in (getattr(s, "words", None) or []):
                    words.append({
                        "start": round(float(w.start), 3) if w.start is not None else None,
                        "end": round(float(w.end), 3) if w.end is not None else None,
                        "word": w.word,
                        "probability": round(float(w.probability), 3) if getattr(w, "probability", None) is not None else None
                    })
                collected_segs.append({
                    "start": round(float(s.start), 3) if s.start is not None else None,
                    "end": round(float(s.end), 3) if s.end is not None else None,
                    "text": s.text.strip(),
                    "avg_logprob": round(float(s.avg_logprob), 3) if getattr(s, "avg_logprob", None) is not None else None,
                    "no_speech_prob": round(float(s.no_speech_prob), 3) if getattr(s, "no_speech_prob", None) is not None else None,
                    "words": words
                })
            return collected_segs, info

        retried_no_vad = False
        try:
            segments, info = _run(use_vad=True)
        except Exception as exc:
            if "no clip timestamps found" in str(exc).lower():
                retried_no_vad = True
                segments, info = _run(use_vad=False)
            else:
                raise

        if not any(s["text"] for s in segments) and not retried_no_vad:
            retried_no_vad = True
            segments, info = _run(use_vad=False)

        full_text = " ".join(s["text"] for s in segments if s["text"]).strip()
        alerts = []
        if retried_no_vad:
            alerts.append("retried_without_vad")
        if not full_text:
            alerts.append("inaudible_or_silent")
        if any(s.get("avg_logprob") is not None and s["avg_logprob"] < -0.85 for s in segments):
            alerts.append("low_confidence_review")

        detected_lang = getattr(info, "language", target_lang)
        lang_prob = round(float(getattr(info, "language_probability", 1.0)), 3)
        duration = round(float(getattr(info, "duration", 0.0)), 2)

        return {
            "text": full_text,
            "duration": duration,
            "language": detected_lang,
            "confidence": lang_prob,
            "segments": segments,
            "alerts": alerts,
            "engine": f"faster-whisper/{self.model_name} ({self.device}:{self.compute_type})"
        }
