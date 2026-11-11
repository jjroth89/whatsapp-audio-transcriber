import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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

@dataclass
class MediaCatalogItem:
    message_index: int
    filename: str
    attachment_type: str
    timestamp: str
    sender: str
    extracted: bool
    local_rel_path: Optional[str] = None

# Matches standard WhatsApp audio filenames like PTT-YYYYMMDD-WAXXXX or AUD-YYYYMMDD-WAXXXX
DATE_IN_FILENAME_REGEX = re.compile(r"(?:PTT|AUD)-(\d{4})(\d{2})(\d{2})-", re.IGNORECASE)

def calculate_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def match_audios_and_catalog_media(
    messages: List[ChatMessage],
    media_dir: Path
) -> Tuple[List[AudioAttachmentMatch], List[MediaCatalogItem]]:
    """
    Finds and matches audio files for transcription, while building a complete
    catalog of all media and documents (images, videos, PDFs) referenced in the chat.
    """
    # Index all available files on disk (case-insensitive lookup)
    disk_index: Dict[str, Path] = {}
    if media_dir.exists():
        for p in media_dir.rglob("*"):
            if p.is_file():
                disk_index[p.name.lower()] = p

    audio_matches: List[AudioAttachmentMatch] = []
    catalog_items: List[MediaCatalogItem] = []

    for idx, msg in enumerate(messages, 1):
        if not msg.attachment_filename:
            continue

        fname = msg.attachment_filename
        fname_lower = fname.lower()
        matched_path = disk_index.get(fname_lower)
        extracted = matched_path is not None
        att_type = msg.attachment_type or "attachment"

        # Add to complete media catalog
        rel_path = f"media/{matched_path.name}" if extracted else None
        catalog_items.append(MediaCatalogItem(
            message_index=idx,
            filename=fname,
            attachment_type=att_type,
            timestamp=f"{msg.date_iso} {msg.time_str}",
            sender=msg.sender,
            extracted=extracted,
            local_rel_path=rel_path
        ))

        # If audio, prepare for transcription
        if att_type == "audio" and extracted:
            sha256_hash = calculate_sha256(matched_path)
            is_shared_or_forwarded = msg.is_forwarded
            origin_date = None
            provenance = "Native voice note recorded in chat"

            date_match = DATE_IN_FILENAME_REGEX.search(fname)
            if date_match:
                y, m, d = date_match.groups()
                origin_date = f"{y}-{m}-{d}"
                if origin_date != msg.date_iso:
                    is_shared_or_forwarded = True
                    provenance = f"Shared/forwarded audio originally recorded on {origin_date}"
            elif fname.upper().startswith("AUD-"):
                is_shared_or_forwarded = True
                provenance = "External audio track (AUD prefix)"

            if msg.is_forwarded and not provenance.startswith("Shared/forwarded"):
                provenance = "Forwarded message (explicit marker in chat)"

            audio_matches.append(AudioAttachmentMatch(
                message=msg,
                filename=matched_path.name,
                file_path=matched_path,
                sha256=sha256_hash,
                is_forwarded_or_shared=is_shared_or_forwarded,
                origin_date=origin_date,
                provenance_note=provenance
            ))

    return audio_matches, catalog_items

def match_audios_in_chat(messages: List[ChatMessage], media_dir: Path) -> List[AudioAttachmentMatch]:
    """Convenience alias returning only the audio matches."""
    audios, _ = match_audios_and_catalog_media(messages, media_dir)
    return audios
