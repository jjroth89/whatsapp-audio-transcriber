import re
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .parser import ChatMessage, classify_attachment
from .vcf import parse_vcf_file


@dataclass
class AudioAttachmentMatch:
    message: ChatMessage
    filename: str
    file_path: Path
    sha256: str
    is_forwarded_or_shared: bool
    origin_date: Optional[str] = None
    provenance_note: str = ""
    media_kind: str = "audio"  # "audio" or "video"
    size_bytes: int = 0


@dataclass
class MediaCatalogItem:
    message_index: int
    filename: str
    attachment_type: str
    timestamp: str
    sender: str
    extracted: bool
    local_rel_path: Optional[str] = None
    saved_filename: Optional[str] = None
    retention_status: str = "retained_original"
    attachment_evidence: Optional[str] = None
    text_extraction: Optional[Dict[str, Any]] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    contact_info: Optional[Dict[str, Any]] = None


# Matches standard WhatsApp audio filenames like PTT-YYYYMMDD-WAXXXX or AUD-YYYYMMDD-WAXXXX
DATE_IN_FILENAME_REGEX = re.compile(r"(?:PTT|AUD|IMG|VID|STK)-(\d{4})(\d{2})(\d{2})-", re.IGNORECASE)
ANY_DATE_PREFIX_REGEX = re.compile(r"(?:^|\D)(?:19|20)\d{2}[-_]?\d{2}[-_]?\d{2}(?:\D|$)")


def calculate_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_time_hhmm(time_str: str) -> str:
    """Convert '14:32', '14:32:05', or '2:32 PM' into '1432'."""
    t = time_str.strip()
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p", "%I:%M%p"):
        try:
            dt = datetime.strptime(t.upper(), fmt)
            return dt.strftime("%H%M")
        except ValueError:
            continue
    digits = re.sub(r"\D", "", t)
    return digits[:4].ljust(4, "0") if digits else "0000"


def build_timestamped_filename(filename: str, date_iso: str, time_str: str) -> str:
    """
    Ensure non-timestamped documents/projects (e.g. PDFs, DWGs, VCFs) receive
    a chronological YYYY-MM-DD_HHMM_ prefix so agents can sort and locate them.
    """
    if DATE_IN_FILENAME_REGEX.search(filename) or ANY_DATE_PREFIX_REGEX.search(filename):
        return filename
    hhmm = _normalize_time_hhmm(time_str)
    safe_date = date_iso if re.match(r"^\d{4}-\d{2}-\d{2}$", date_iso) else "0000-00-00"
    return f"{safe_date}_{hhmm}_{filename}"


def timestamp_retained_file(
    matched_path: Path,
    msg: ChatMessage,
    media_dir: Path,
    used_names: Set[str],
) -> Path:
    """
    Rename a retained non-AV file in media_dir to include its message timestamp
    if it lacks one, avoiding filename collisions.
    """
    desired_name = build_timestamped_filename(matched_path.name, msg.date_iso, msg.time_str)
    candidate = desired_name
    stem = Path(desired_name).stem
    suffix = Path(desired_name).suffix
    counter = 2
    while candidate.lower() in used_names and candidate.lower() != matched_path.name.lower():
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1

    target_path = media_dir / candidate
    if matched_path.resolve() != target_path.resolve():
        # Only rename if inside media_dir (do not mutate user's input directory when input is a folder)
        if matched_path.parent.resolve() == media_dir.resolve() and not target_path.exists():
            matched_path.rename(target_path)
            used_names.add(target_path.name.lower())
            return target_path

    used_names.add(matched_path.name.lower())
    return matched_path


def match_audios_and_catalog_media(
    messages: List[ChatMessage],
    media_dir: Path,
    include_videos: bool = False,
    available_in_source: Optional[Set[str]] = None,
    timestamp_docs: bool = True,
    rename_on_disk: bool = True,
) -> Tuple[List[AudioAttachmentMatch], List[MediaCatalogItem]]:
    """
    Finds and matches audio (and optionally video) files for transcription,
    parses .vcf contacts, timestamps non-timestamped documents/CAD files,
    and builds a complete catalog of all media and documents referenced in the chat.
    """
    disk_index: Dict[str, Path] = {}
    if media_dir.exists():
        for p in sorted(media_dir.rglob("*"), key=lambda item: item.as_posix().casefold()):
            if p.is_file():
                key = p.name.casefold()
                if key in disk_index and disk_index[key] != p:
                    raise ValueError(f"Ambiguous media basename: {p.name}")
                disk_index[key] = p

    used_names: Set[str] = {p.name.lower() for p in disk_index.values()}
    audio_matches: List[AudioAttachmentMatch] = []
    catalog_items: List[MediaCatalogItem] = []
    referenced_names: Set[str] = set()

    for idx, msg in enumerate(messages, 1):
        if not msg.attachment_filename:
            continue

        fname = msg.attachment_filename
        fname_lower = fname.casefold()
        referenced_names.add(fname_lower)
        matched_path = disk_index.get(fname_lower)
        extracted = matched_path is not None
        att_type = msg.attachment_type or "attachment"

        sha256_hash: Optional[str] = None
        size_bytes: Optional[int] = None
        contact_info: Optional[Dict[str, Any]] = None
        saved_name: Optional[str] = matched_path.name if matched_path else None

        if extracted and matched_path is not None:
            size_bytes = matched_path.stat().st_size
            sha256_hash = calculate_sha256(matched_path)

            if att_type == "contact":
                try:
                    contact_info = parse_vcf_file(matched_path)
                except Exception:
                    contact_info = None

            # Timestamp non-AV attachments lacking a date in their filename
            if timestamp_docs and rename_on_disk and att_type not in ("audio", "video", "sticker"):
                updated_path = timestamp_retained_file(matched_path, msg, media_dir, used_names)
                if updated_path != matched_path:
                    disk_index[fname_lower] = updated_path
                    matched_path = updated_path
                    saved_name = updated_path.name

            retention_status = "retained_original"
            rel_path = f"media/{saved_name}"
        else:
            rel_path = None
            if available_in_source is not None and fname_lower in available_in_source:
                retention_status = "not_packaged_by_choice"
            else:
                retention_status = "source_reference_unavailable"

        catalog_items.append(MediaCatalogItem(
            message_index=idx,
            filename=fname,
            saved_filename=saved_name,
            attachment_type=att_type,
            timestamp=f"{msg.date_iso} {msg.time_str}",
            sender=msg.sender,
            extracted=extracted,
            retention_status=retention_status,
            sha256=sha256_hash,
            size_bytes=size_bytes,
            local_rel_path=rel_path,
            contact_info=contact_info,
            attachment_evidence=msg.attachment_evidence,
        ))

        should_transcribe = (att_type == "audio") or (include_videos and att_type == "video")
        if should_transcribe and extracted and matched_path is not None:
            is_shared_or_forwarded = msg.is_forwarded
            origin_date = None
            if att_type == "video":
                provenance = "Video audio track extracted for transcription"
            else:
                provenance = "Native voice note recorded in chat"

            date_match = DATE_IN_FILENAME_REGEX.search(fname)
            if date_match:
                y, m, d = date_match.groups()
                origin_date = f"{y}-{m}-{d}"
                if origin_date != msg.date_iso:
                    is_shared_or_forwarded = True
                    provenance = f"Shared/forwarded {att_type} originally recorded on {origin_date}"
            elif fname.upper().startswith("AUD-"):
                is_shared_or_forwarded = True
                provenance = "External audio track (AUD prefix)"

            if msg.is_forwarded and not provenance.startswith("Shared/forwarded"):
                provenance = "Forwarded message (explicit marker in chat)"

            audio_matches.append(AudioAttachmentMatch(
                message=msg,
                filename=matched_path.name,
                file_path=matched_path,
                sha256=sha256_hash or calculate_sha256(matched_path),
                is_forwarded_or_shared=is_shared_or_forwarded,
                origin_date=origin_date,
                provenance_note=provenance,
                media_kind=att_type,
                size_bytes=size_bytes or matched_path.stat().st_size,
            ))

    # Retained unreferenced source assets belong in the corpus index, but are not
    # invented WhatsApp messages (message_index=0, timestamp unknown).
    for key, path in disk_index.items():
        if key in referenced_names or not path.exists():
            continue
        att_type = classify_attachment(path.name)
        if att_type not in ("document", "cad", "contact", "image"):
            continue
        info = None
        if att_type == "contact":
            try:
                info = parse_vcf_file(path)
            except (OSError, ValueError, UnicodeError):
                pass
        catalog_items.append(MediaCatalogItem(
            message_index=0,
            filename=path.name,
            attachment_type=att_type,
            timestamp="",
            sender="",
            extracted=True,
            local_rel_path=f"media/{path.name}",
            saved_filename=path.name,
            retention_status="retained_original",
            sha256=calculate_sha256(path),
            size_bytes=path.stat().st_size,
            contact_info=info,
        ))
    return audio_matches, catalog_items


def match_audios_in_chat(messages: List[ChatMessage], media_dir: Path) -> List[AudioAttachmentMatch]:
    """Convenience alias returning only the audio matches."""
    audios, _ = match_audios_and_catalog_media(messages, media_dir, rename_on_disk=False)
    return audios
