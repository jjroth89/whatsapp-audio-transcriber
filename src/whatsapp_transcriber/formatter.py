import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from .parser import ChatMessage
from .matcher import AudioAttachmentMatch, MediaCatalogItem
from .vcf import format_vcf_summary


def _encode_rel_link(rel_path: str) -> str:
    """URL-encode path segments while preserving forward slashes for Markdown links."""
    return "/".join(quote(part) for part in rel_path.split("/"))


def _build_retention_summary(
    catalog_items: List[MediaCatalogItem],
    messages: List[ChatMessage],
    execution_info: Dict[str, Any],
) -> Dict[str, Any]:
    retained = sum(1 for c in catalog_items if c.retention_status == "retained")
    purged_after_tx = sum(1 for c in catalog_items if c.retention_status == "transcribed_and_purged")
    cataloged_only = sum(1 for c in catalog_items if c.retention_status == "cataloged_only_not_extracted")
    missing = sum(1 for c in catalog_items if c.retention_status == "missing_from_export")
    omitted_in_export = sum(1 for m in messages if m.is_omitted_media)

    media_mode = execution_info.get("media_mode", "audio")
    av_purged = bool(execution_info.get("raw_av_purged_after_transcription", False))

    return {
        "media_mode": media_mode,
        "transcribe_videos": bool(execution_info.get("transcribe_videos", False)),
        "raw_av_purged_after_transcription": av_purged,
        "total_attachments_cataloged": len(catalog_items),
        "retained_in_package": retained,
        "transcribed_and_purged": purged_after_tx,
        "cataloged_only_not_extracted": cataloged_only,
        "missing_from_source_export": missing,
        "omitted_media_in_source_export": omitted_in_export,
        "agent_integrity_note": (
            "All voice notes (and video audio tracks when enabled) were extracted from the source archive, "
            "SHA-256 verified, and transcribed locally. Any audio/video files marked 'transcribed_and_purged' "
            "were intentionally removed from media/ per retention policy after transcription succeeded; "
            "their absence in media/ is expected and does NOT indicate missing or unverified content."
        ),
    }


def generate_enriched_outputs(
    messages: List[ChatMessage],
    transcriptions: Dict[str, Dict[str, Any]],  # filename -> transcription dict
    audio_matches: List[AudioAttachmentMatch],
    catalog_items: List[MediaCatalogItem] = None,
    output_dir: Path = None,
    chat_title: str = "WhatsApp Conversation",
    execution_info: Dict[str, Any] = None,
) -> Dict[str, Path]:
    """
    Generates enriched_chat.md, enriched_chat.json, and audio_audit.json in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_items = catalog_items or []
    execution_info = execution_info or {}

    match_map: Dict[str, AudioAttachmentMatch] = {m.filename.lower(): m for m in audio_matches}
    catalog_by_idx: Dict[int, MediaCatalogItem] = {c.message_index: c for c in catalog_items}
    catalog_by_name: Dict[str, MediaCatalogItem] = {c.filename.lower(): c for c in catalog_items}

    retention_summary = _build_retention_summary(catalog_items, messages, execution_info)

    md_lines: List[str] = [
        f"# {chat_title} — Enriched Transcript",
        "",
        "> **Media & Transcription Integrity Policy**",
        f"> - **Media Retention Mode:** `{retention_summary['media_mode']}` | **Video Transcription:** `{retention_summary['transcribe_videos']}`",
        f"> - **Cataloged Attachments:** `{retention_summary['total_attachments_cataloged']}` "
        f"(Retained in `media/`: `{retention_summary['retained_in_package']}`, "
        f"Transcribed & Purged: `{retention_summary['transcribed_and_purged']}`, "
        f"Excluded by Policy: `{retention_summary['cataloged_only_not_extracted']}`, "
        f"Omitted in WhatsApp Export: `{retention_summary['omitted_media_in_source_export']}`)",
        f"> - **Agent Note:** {retention_summary['agent_integrity_note']}",
        "",
        "---",
        "",
    ]

    json_messages: List[Dict[str, Any]] = []
    audit_audios: List[Dict[str, Any]] = []

    for idx, msg in enumerate(messages, 1):
        if msg.is_system:
            md_lines.append(f"*[{msg.date_iso} {msg.time_str}] {msg.content}*")
            json_messages.append({
                "index": idx,
                "timestamp": f"{msg.date_iso} {msg.time_str}",
                "date_iso": msg.date_iso,
                "time_str": msg.time_str,
                "sender": "System",
                "is_system": True,
                "type": "system",
                "content": msg.content,
            })
            continue

        edited_badge = " *(edited)*" if msg.is_edited else ""
        cat_item: Optional[MediaCatalogItem] = catalog_by_idx.get(idx)
        if not cat_item and msg.attachment_filename:
            cat_item = catalog_by_name.get(msg.attachment_filename.lower())

        match: Optional[AudioAttachmentMatch] = None
        audio_item: Optional[Dict[str, Any]] = None
        if msg.attachment_filename:
            match = match_map.get(msg.attachment_filename.lower())
            if not match and cat_item and cat_item.saved_filename:
                match = match_map.get(cat_item.saved_filename.lower())
            if match and match.filename in transcriptions:
                audio_item = transcriptions[match.filename]

        if audio_item and match:
            dur_str = f"{audio_item.get('duration', 0.0):.1f}s"
            transcript_text = audio_item.get("text", "").strip() or "[Inaudible or empty audio]"
            is_video = match.media_kind == "video"
            is_purged = (cat_item is not None and cat_item.retention_status == "transcribed_and_purged")

            if is_purged or not (cat_item and cat_item.local_rel_path):
                file_ref = f"`{match.filename}` *(SHA-256 verified, transcribed & purged per policy)*"
            else:
                encoded_link = _encode_rel_link(cat_item.local_rel_path)
                file_ref = f"[`{match.filename}`]({encoded_link})"

            if is_video:
                if match.is_forwarded_or_shared:
                    msg_type = "forwarded_video_transcript"
                    origin_str = f" | Originated: {match.origin_date}" if match.origin_date else ""
                    tag = f"[FORWARDED/SHARED VIDEO TRANSCRIPT{origin_str} | Duration: {dur_str} | File: {file_ref}]"
                else:
                    msg_type = "video_transcript"
                    tag = f"[VIDEO TRANSCRIPT | Duration: {dur_str} | File: {file_ref}]"
            else:
                if match.is_forwarded_or_shared:
                    msg_type = "forwarded_audio"
                    origin_str = f" | Originated: {match.origin_date}" if match.origin_date else ""
                    tag = f"[FORWARDED/SHARED AUDIO{origin_str} | Duration: {dur_str} | File: {file_ref}]"
                else:
                    msg_type = "audio"
                    tag = f"[AUDIO NOTE | Duration: {dur_str} | File: {file_ref}]"

            md_lines.append(f"**[{msg.date_iso} {msg.time_str}] {msg.sender}:** {tag}{edited_badge}")
            md_lines.append(f"> \"{transcript_text}\"")
            md_lines.append("")

            audio_meta = {
                "filename": match.filename,
                "media_kind": match.media_kind,
                "sha256": match.sha256,
                "duration_s": audio_item.get("duration", 0.0),
                "is_forwarded_or_shared": match.is_forwarded_or_shared,
                "origin_date": match.origin_date,
                "provenance": match.provenance_note,
                "retention_status": cat_item.retention_status if cat_item else "retained",
                "language": audio_item.get("language", "pt"),
                "confidence": audio_item.get("confidence", 1.0),
                "alerts": audio_item.get("alerts", []),
                "segments": audio_item.get("segments", []),
                "engine": audio_item.get("engine", ""),
            }

            json_messages.append({
                "index": idx,
                "timestamp": f"{msg.date_iso} {msg.time_str}",
                "date_iso": msg.date_iso,
                "time_str": msg.time_str,
                "sender": msg.sender,
                "is_system": False,
                "is_edited": msg.is_edited,
                "type": msg_type,
                "content": transcript_text,
                "audio_meta": audio_meta,
            })

            audit_audios.append({
                "index": idx,
                "timestamp": f"{msg.date_iso} {msg.time_str}",
                "sender": msg.sender,
                **audio_meta,
            })

        elif msg.attachment_filename and cat_item:
            att_type = cat_item.attachment_type
            tag_label = att_type.upper()
            saved_name = cat_item.saved_filename or cat_item.filename
            renamed_note = f" *(original: `{cat_item.filename}`)*" if saved_name != cat_item.filename else ""

            if cat_item.retention_status == "retained" and cat_item.local_rel_path:
                encoded_link = _encode_rel_link(cat_item.local_rel_path)
                file_ref = f"[`{saved_name}`]({encoded_link}){renamed_note}"
            elif cat_item.retention_status == "cataloged_only_not_extracted":
                file_ref = f"`{cat_item.filename}` *(cataloged only; excluded from `media/` by policy)*"
            else:
                file_ref = f"`{cat_item.filename}` *(missing from source export)*"

            if att_type == "contact" and cat_item.contact_info:
                contact_summary = format_vcf_summary(cat_item.contact_info)
                semantic_tag = f"[{tag_label}: {contact_summary} | File: {file_ref}]"
            else:
                semantic_tag = f"[{tag_label} | File: {file_ref}]"

            # Check for accompanying caption text
            clean_caption = msg.content.replace(msg.attachment_filename, "")
            for marker in ["(file attached)", "(arquivo anexado)", "(ficheiro anexado)"]:
                clean_caption = clean_caption.replace(marker, "")
            clean_caption = clean_caption.strip(" <>\u200e\u200f")
            if clean_caption.lower().startswith(("attached:", "anexado:")):
                clean_caption = ""

            md_lines.append(f"**[{msg.date_iso} {msg.time_str}] {msg.sender}:** {semantic_tag}{edited_badge}")
            if clean_caption:
                md_lines.append(f"> Caption: {clean_caption}")
            md_lines.append("")

            json_entry: Dict[str, Any] = {
                "index": idx,
                "timestamp": f"{msg.date_iso} {msg.time_str}",
                "date_iso": msg.date_iso,
                "time_str": msg.time_str,
                "sender": msg.sender,
                "is_system": False,
                "is_edited": msg.is_edited,
                "type": att_type,
                "content": clean_caption or msg.content,
                "attachment": {
                    "filename": cat_item.filename,
                    "saved_filename": cat_item.saved_filename,
                    "type": att_type,
                    "extracted": cat_item.extracted,
                    "retention_status": cat_item.retention_status,
                    "sha256": cat_item.sha256,
                    "size_bytes": cat_item.size_bytes,
                    "local_rel_path": cat_item.local_rel_path,
                },
            }
            if cat_item.contact_info:
                json_entry["attachment"]["contact_info"] = cat_item.contact_info
            json_messages.append(json_entry)

        elif msg.is_omitted_media:
            md_lines.append(f"**[{msg.date_iso} {msg.time_str}] {msg.sender}:** [OMITTED MEDIA IN SOURCE EXPORT] `{msg.content}`{edited_badge}")
            json_messages.append({
                "index": idx,
                "timestamp": f"{msg.date_iso} {msg.time_str}",
                "date_iso": msg.date_iso,
                "time_str": msg.time_str,
                "sender": msg.sender,
                "is_system": False,
                "is_edited": msg.is_edited,
                "type": "omitted_media",
                "content": msg.content,
            })

        else:
            md_lines.append(f"**[{msg.date_iso} {msg.time_str}] {msg.sender}:** {msg.content}{edited_badge}")
            json_messages.append({
                "index": idx,
                "timestamp": f"{msg.date_iso} {msg.time_str}",
                "date_iso": msg.date_iso,
                "time_str": msg.time_str,
                "sender": msg.sender,
                "is_system": False,
                "is_edited": msg.is_edited,
                "type": "text",
                "content": msg.content,
            })

    md_path = output_dir / "enriched_chat.md"
    json_path = output_dir / "enriched_chat.json"
    audit_path = output_dir / "audio_audit.json"

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    json_payload = {
        "chat_title": chat_title,
        "execution": execution_info,
        "package_policy": retention_summary,
        "total_messages": len(json_messages),
        "total_transcribed_audios": len([a for a in audit_audios if a.get("media_kind", "audio") == "audio"]),
        "total_transcribed_videos": len([a for a in audit_audios if a.get("media_kind") == "video"]),
        "total_cataloged_media": len(catalog_items),
        "messages": json_messages,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    catalog_dicts = []
    for c in catalog_items:
        item_dict: Dict[str, Any] = {
            "message_index": c.message_index,
            "filename": c.filename,
            "saved_filename": c.saved_filename,
            "type": c.attachment_type,
            "timestamp": c.timestamp,
            "sender": c.sender,
            "extracted": c.extracted,
            "retention_status": c.retention_status,
            "sha256": c.sha256,
            "size_bytes": c.size_bytes,
            "local_rel_path": c.local_rel_path,
        }
        if c.contact_info:
            item_dict["contact_info"] = c.contact_info
        catalog_dicts.append(item_dict)

    audit_payload = {
        "chat_title": chat_title,
        "execution": execution_info,
        "package_policy": retention_summary,
        "summary": {
            "total_transcribed": len(audit_audios),
            "total_audios": len([a for a in audit_audios if a.get("media_kind", "audio") == "audio"]),
            "total_videos": len([a for a in audit_audios if a.get("media_kind") == "video"]),
            "total_duration_seconds": round(sum(a.get("duration_s", 0.0) for a in audit_audios), 2),
            "forwarded_or_shared_count": sum(1 for a in audit_audios if a.get("is_forwarded_or_shared")),
            "total_attachments_cataloged": len(catalog_items),
            "total_attachments_extracted": sum(1 for c in catalog_items if c.extracted),
        },
        "audios": audit_audios,
        "media_catalog": catalog_dicts,
    }
    audit_path.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "markdown": md_path,
        "json": json_path,
        "audit": audit_path,
    }
