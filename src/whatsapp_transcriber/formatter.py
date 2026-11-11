import json
from pathlib import Path
from typing import Dict, List, Any
from .parser import ChatMessage
from .matcher import AudioAttachmentMatch, MediaCatalogItem

def generate_enriched_outputs(
    messages: List[ChatMessage],
    transcriptions: Dict[str, Dict[str, Any]],  # filename -> transcription dict
    audio_matches: List[AudioAttachmentMatch],
    catalog_items: List[MediaCatalogItem] = None,
    output_dir: Path = None,
    chat_title: str = "WhatsApp Conversation",
    execution_info: Dict[str, Any] = None
) -> Dict[str, Path]:
    """
    Generates enriched_chat.md, enriched_chat.json, and audio_audit.json in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_items = catalog_items or []
    match_map: Dict[str, AudioAttachmentMatch] = {m.filename.lower(): m for m in audio_matches}
    catalog_map: Dict[str, MediaCatalogItem] = {c.filename.lower(): c for c in catalog_items}

    md_lines: List[str] = [
        f"# {chat_title} — Enriched Transcript",
        "",
        "> Generated automatically by whatsapp-audio-transcriber.",
        "> Audio voice notes have been transcribed and embedded chronologically into the conversation timeline.",
        "> Non-audio media and documents are semantically tagged.",
        "",
        "---",
        ""
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
                "content": msg.content
            })
            continue

        audio_item = None
        cat_item = None
        if msg.attachment_filename:
            fname_lower = msg.attachment_filename.lower()
            cat_item = catalog_map.get(fname_lower)
            match = match_map.get(fname_lower)
            if match and match.filename in transcriptions:
                audio_item = transcriptions[match.filename]

        if audio_item:
            match = match_map[msg.attachment_filename.lower()]
            dur_str = f"{audio_item.get('duration', 0.0):.1f}s"
            transcript_text = audio_item.get("text", "").strip() or "[Inaudible or empty audio]"

            link_target = f"media/{match.filename}"
            if match.is_forwarded_or_shared:
                msg_type = "forwarded_audio"
                origin_str = f" | Originated: {match.origin_date}" if match.origin_date else ""
                tag = f"[FORWARDED/SHARED AUDIO{origin_str} | Duration: {dur_str} | File: [`{match.filename}`]({link_target})]"
            else:
                msg_type = "audio"
                tag = f"[AUDIO NOTE | Duration: {dur_str} | File: [`{match.filename}`]({link_target})]"

            md_lines.append(f"**[{msg.date_iso} {msg.time_str}] {msg.sender}:** {tag}")
            md_lines.append(f"> \"{transcript_text}\"")
            md_lines.append("")

            audio_meta = {
                "filename": match.filename,
                "sha256": match.sha256,
                "duration_s": audio_item.get("duration", 0.0),
                "is_forwarded_or_shared": match.is_forwarded_or_shared,
                "origin_date": match.origin_date,
                "provenance": match.provenance_note,
                "language": audio_item.get("language", "pt"),
                "confidence": audio_item.get("confidence", 1.0),
                "alerts": audio_item.get("alerts", []),
                "segments": audio_item.get("segments", []),
                "engine": audio_item.get("engine", "")
            }

            json_messages.append({
                "index": idx,
                "timestamp": f"{msg.date_iso} {msg.time_str}",
                "date_iso": msg.date_iso,
                "time_str": msg.time_str,
                "sender": msg.sender,
                "is_system": False,
                "type": msg_type,
                "content": transcript_text,
                "audio_meta": audio_meta
            })

            audit_audios.append({
                "message_index": idx,
                "timestamp": f"{msg.date_iso} {msg.time_str}",
                "sender": msg.sender,
                **audio_meta
            })
        elif cat_item:
            # Non-audio attachment (image, video, document, sticker)
            fname = cat_item.filename
            att_type = cat_item.attachment_type.upper()
            file_link = f"[`{fname}`]({cat_item.local_rel_path})" if cat_item.extracted else f"`{fname}` (not extracted)"

            md_lines.append(f"**[{msg.date_iso} {msg.time_str}] {msg.sender}:** [{att_type} ATTACHMENT: {file_link}]")
            json_messages.append({
                "index": idx,
                "timestamp": f"{msg.date_iso} {msg.time_str}",
                "date_iso": msg.date_iso,
                "time_str": msg.time_str,
                "sender": msg.sender,
                "is_system": False,
                "type": cat_item.attachment_type,
                "content": f"[{att_type} ATTACHMENT: {fname}]",
                "attachment_meta": {
                    "filename": fname,
                    "type": cat_item.attachment_type,
                    "extracted": cat_item.extracted,
                    "path": cat_item.local_rel_path
                }
            })
        else:
            # Standard text message
            md_lines.append(f"**[{msg.date_iso} {msg.time_str}] {msg.sender}:** {msg.content}")
            json_messages.append({
                "index": idx,
                "timestamp": f"{msg.date_iso} {msg.time_str}",
                "date_iso": msg.date_iso,
                "time_str": msg.time_str,
                "sender": msg.sender,
                "is_system": False,
                "type": "text",
                "content": msg.content
            })

    # File paths
    md_path = output_dir / "enriched_chat.md"
    json_path = output_dir / "enriched_chat.json"
    audit_path = output_dir / "audio_audit.json"

    # Write Markdown
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Write Enriched JSON
    json_path.write_text(json.dumps(json_messages, ensure_ascii=False, indent=2), encoding="utf-8")

    # Catalog summary
    media_summary = {
        "total_referenced_attachments": len(catalog_items),
        "total_extracted_attachments": sum(1 for c in catalog_items if c.extracted),
        "by_type": {
            "audio": sum(1 for c in catalog_items if c.attachment_type == "audio"),
            "image": sum(1 for c in catalog_items if c.attachment_type == "image"),
            "video": sum(1 for c in catalog_items if c.attachment_type == "video"),
            "document": sum(1 for c in catalog_items if c.attachment_type == "document"),
            "sticker": sum(1 for c in catalog_items if c.attachment_type == "sticker")
        },
        "items": [
            {
                "message_index": c.message_index,
                "filename": c.filename,
                "type": c.attachment_type,
                "timestamp": c.timestamp,
                "sender": c.sender,
                "extracted": c.extracted,
                "rel_path": c.local_rel_path
            }
            for c in catalog_items
        ]
    }

    # Write Audit Report
    audit_report = {
        "chat_title": chat_title,
        "execution_info": execution_info or {},
        "total_messages": len(messages),
        "total_audios_transcribed": len(audit_audios),
        "media_catalog": media_summary,
        "audios": audit_audios
    }
    audit_path.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "markdown": md_path,
        "json": json_path,
        "audit": audit_path
    }
