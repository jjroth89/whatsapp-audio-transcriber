import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from .parser import ChatMessage

@dataclass
class AudioAttachmentMatch:
    message: ChatMessage
    filename: str
    file_path: Path
    sha256: str
    is_forwarded_or_shared: bool
    origin_date: Optional[str] = None
    provenance_note: str = ""

# Matches standard WhatsApp audio filenames like PTT-YYYYMMDD-WAXXXX or AUD-YYYYMMDD-WAXXXX
DATE_IN_FILENAME_REGEX = re.compile(r"(?:PTT|AUD)-(\d{4})(\d{2})(\d{2})-", re.IGNORECASE)

def calculate_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def match_audios_in_chat(
    messages: List[ChatMessage],
    audio_files_dir: Path
) -> List[AudioAttachmentMatch]:
    """
    Finds all audio files referenced in messages, verifies their existence on disk,
    checks provenance (native voice note vs shared/forwarded), and computes SHA-256 hashes.
    """
    # Index available audio files in the folder (case-insensitive filename lookup)
    audio_index: Dict[str, Path] = {}
    for p in audio_files_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".opus", ".ogg", ".m4a", ".mp3", ".wav", ".aac", ".flac"]:
            audio_index[p.name.lower()] = p

    matches: List[AudioAttachmentMatch] = []

    for msg in messages:
        if not msg.attachment_filename:
            continue

        fname = msg.attachment_filename
        fname_lower = fname.lower()
        matched_path = audio_index.get(fname_lower)

        if not matched_path:
            # File referenced in chat but not found in media folder
            continue

        sha256_hash = calculate_sha256(matched_path)

        # Check provenance: native PTT voice note vs shared / forwarded
        is_shared_or_forwarded = msg.is_forwarded
        origin_date = None
        provenance = "Native voice note recorded in chat"

        date_match = DATE_IN_FILENAME_REGEX.search(fname)
        if date_match:
            y, m, d = date_match.groups()
            origin_date = f"{y}-{m}-{d}"
            # If the filename date does not match the chat message date, it was created elsewhere
            if origin_date != msg.date_iso:
                is_shared_or_forwarded = True
                provenance = f"Shared/forwarded audio originally recorded on {origin_date}"
        elif fname.upper().startswith("AUD-"):
            is_shared_or_forwarded = True
            provenance = "External audio track (AUD prefix)"

        if msg.is_forwarded and not provenance.startswith("Shared/forwarded"):
            provenance = "Forwarded message (explicit marker in chat)"

        matches.append(AudioAttachmentMatch(
            message=msg,
            filename=matched_path.name,
            file_path=matched_path,
            sha256=sha256_hash,
            is_forwarded_or_shared=is_shared_or_forwarded,
            origin_date=origin_date,
            provenance_note=provenance
        ))

    return matches
