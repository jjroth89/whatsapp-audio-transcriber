import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from .parser import ChatMessage
from .matcher import AudioAttachmentMatch, MediaCatalogItem
from .vcf import format_vcf_summary
from .ai_view import annotate_ai_payload, write_ai_markdown


def _encode_rel_link(rel_path: str) -> str:
    """URL-encode path segments while preserving forward slashes for Markdown links."""
    return "/".join(quote(part) for part in rel_path.split("/"))


def _verified_transcript(result: Optional[Dict[str, Any]]) -> bool:
    if not result or not str(result.get("text", "")).strip():
        return False
    return not any(alert in result.get("alerts", [])
                   for alert in ("no_audio_track", "inaudible_or_silent", "processing_failed"))


def _build_retention_summary(
    catalog_items: List[MediaCatalogItem],
    messages: List[ChatMessage],
    execution_info: Dict[str, Any],
    audio_matches: List[AudioAttachmentMatch],
    transcriptions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    categories = (
        "retained_original", "text_equivalent_verified", "not_packaged_by_choice",
        "source_reference_unavailable", "decorative_stickers_skipped", "processing_failed",
    )
    counts = {key: 0 for key in categories}
    for item in catalog_items:
        if item.retention_status in counts:
            counts[item.retention_status] += 1

    transcribed_messages = {
        id(m.message) for m in audio_matches
        if _verified_transcript(transcriptions.get(m.filename))
    }
    media_refs = [
        m for m in messages
        if m.attachment_type == "audio"
        or (m.attachment_type == "video" and execution_info.get("transcribe_videos"))
    ]
    verified_count = sum(id(m) in transcribed_messages for m in media_refs)
    unavailable = counts["source_reference_unavailable"]
    omitted_in_source = sum(m.is_omitted_media for m in messages)
    ocr_failures = sum(1 for item in catalog_items if item.text_extraction
                       and item.text_extraction.get("status") == "failed")
    pipeline_complete = (verified_count == len(media_refs) and unavailable == 0 and ocr_failures == 0)
    complete = pipeline_complete and (omitted_in_source == 0)

    partial_reasons: List[str] = []
    if omitted_in_source > 0:
        partial_reasons.append(
            f"{omitted_in_source} <Media omitted> placeholder(s) were already omitted by WhatsApp in the source export"
        )
    if verified_count < len(media_refs):
        partial_reasons.append(
            f"{len(media_refs) - verified_count} audio/video item(s) were silent/inaudible or unverified"
        )
    if unavailable > 0:
        partial_reasons.append(
            f"{unavailable} explicit attachment reference(s) were not present in the source archive"
        )
    if ocr_failures > 0:
        partial_reasons.append(
            f"{ocr_failures} OCR extraction(s) failed"
        )

    return {
        "schema_version": 2,
        "media_mode": execution_info.get("media_mode", "audio"),
        "transcribe_videos": bool(execution_info.get("transcribe_videos", False)),
        "evidence_categories": counts,
        "complete_text_corpus": {
            "scope": "supplied_whatsapp_export",
            "status": "complete" if complete else "partial",
            "pipeline_processing_status": "complete" if pipeline_complete else "partial",
            "partial_reasons": partial_reasons,
            "chat_text_preserved": True,
            "audio_references_total": len(media_refs),
            "audio_text_equivalents_verified": verified_count,
            "unavailable_source_references": unavailable,
            "source_media_placeholders": omitted_in_source,
            "ocr_processing_failures": ocr_failures,
        },
        "agent_integrity_note": (
            "The conversation text represents the supplied export. Available audio/video "
            "text equivalents are individually verified where indicated. Retained PDF, "
            "CAD and image originals are linked; their contents are only text-extracted "
            "when an individual text_extraction result says so. If status is 'partial' while "
            "pipeline_processing_status is 'complete', all supplied files were processed and "
            "'partial' solely reflects WhatsApp <Media omitted> placeholders absent from the "
            "original export (see partial_reasons)."
        ),
    }


def _escape_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("[", "\\[").replace("]", "\\]")


def _build_corpus_index(catalog_items: List[MediaCatalogItem]) -> List[str]:
    indexable = [
        c for c in catalog_items
        if c.attachment_type in ("document", "cad", "contact", "image")
        and c.retention_status == "retained_original" and c.local_rel_path
    ]
    indexable.sort(key=lambda c: (c.timestamp or "9999", c.message_index, (c.saved_filename or c.filename).casefold()))
    lines = ["## Document, Contact & Photo Index", "",
             "| Timestamp | Category | Asset |", "|---|---|---|"]
    labels = {"document": "Document", "cad": "CAD", "contact": "Contact", "image": "Photo"}
    for item in indexable:
        label = item.saved_filename or item.filename
        url = _encode_rel_link(item.local_rel_path)
        lines.append(f"| {_escape_cell(item.timestamp or 'Unreferenced')} "
                     f"| {labels[item.attachment_type]} "
                     f"| [{_escape_cell(label)}]({url}) |")
    if not indexable:
        lines.append("| — | — | No retained documents, contacts or photos |")
    lines.extend(["", "---", ""])
    return lines


def generate_enriched_outputs(
    messages: List[ChatMessage],
    transcriptions: Dict[str, Dict[str, Any]],  # filename -> transcription dict
    audio_matches: List[AudioAttachmentMatch],
    catalog_items: List[MediaCatalogItem] = None,
    output_dir: Path = None,
    chat_title: str = "WhatsApp Conversation",
    execution_info: Dict[str, Any] = None,
    ai_view: bool = False,
    source_sha256: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Generates enriched_chat.md, enriched_chat.json, and audio_audit.json in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_items = catalog_items or []
    execution_info = execution_info or {}

    match_map: Dict[int, AudioAttachmentMatch] = {id(m.message): m for m in audio_matches}
    catalog_by_idx: Dict[int, MediaCatalogItem] = {c.message_index: c for c in catalog_items}
    catalog_by_name: Dict[str, MediaCatalogItem] = {c.filename.lower(): c for c in catalog_items}

    retention_summary = _build_retention_summary(catalog_items, messages, execution_info, audio_matches, transcriptions)

    corpus = retention_summary["complete_text_corpus"]
    reasons_note = f" — {'; '.join(corpus['partial_reasons'])}" if corpus.get("partial_reasons") else ""
    md_lines: List[str] = [
        f"# {chat_title} — Enriched Transcript",
        "",
        f"> **Complete Text Corpus:** `{corpus['status']}` "
        f"(Pipeline Processing: `{corpus['pipeline_processing_status']}`{reasons_note})",
        f"> - Chat messages: preserved | Audio/Video text equivalents verified: "
        f"{corpus['audio_text_equivalents_verified']}/{corpus['audio_references_total']}",
        f"> - WhatsApp `<Media omitted>` placeholders in source: {corpus['source_media_placeholders']} | "
        f"Unavailable explicit references: {corpus['unavailable_source_references']} | "
        f"OCR failures: {corpus['ocr_processing_failures']}",
        f"> - **Ingestion contract:** {retention_summary['agent_integrity_note']}",
        "",
    ] + _build_corpus_index(catalog_items)

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
            match = match_map.get(id(msg))
            if match and match.filename in transcriptions:
                audio_item = transcriptions[match.filename]

        if audio_item and match:
            dur_str = f"{audio_item.get('duration', 0.0):.1f}s"
            transcript_text = audio_item.get("text", "").strip() or "[Inaudible or empty audio]"
            is_video = match.media_kind == "video"
            is_text_only = (cat_item is not None and cat_item.retention_status == "text_equivalent_verified")

            if is_text_only or not (cat_item and cat_item.local_rel_path):
                qualifier = "text equivalent verified" if _verified_transcript(audio_item) else "text extraction unverified"
                file_ref = f"`{match.filename}` *({qualifier}; original not included in pack)*"
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
                "retention_status": cat_item.retention_status if cat_item else "retained_original",
                "text_equivalent_verified": _verified_transcript(audio_item),
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

            if cat_item.retention_status == "retained_original" and cat_item.local_rel_path:
                encoded_link = _encode_rel_link(cat_item.local_rel_path)
                file_ref = f"[`{saved_name}`]({encoded_link}){renamed_note}"
            elif cat_item.retention_status == "not_packaged_by_choice":
                file_ref = f"`{cat_item.filename}` *(original not included by packaging choice)*"
            elif cat_item.retention_status == "source_reference_unavailable":
                file_ref = f"`{cat_item.filename}` *(referenced but not supplied in source)*"
            else:
                file_ref = f"`{cat_item.filename}` *(processing not verified)*"

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
            extraction = cat_item.text_extraction
            if extraction:
                status = extraction.get("status")
                if status == "extracted":
                    full_text = str(extraction.get("text", "")).strip()
                    excerpt = " ".join(full_text.split())
                    truncated = len(excerpt) > 800 or extraction.get("truncated", False)
                    excerpt = excerpt[:800] if len(excerpt) > 800 else excerpt
                    excerpt = excerpt.replace(">", "\\>").replace('"', "\\\"")
                    suffix = "… (full extraction in enriched_chat.json)" if truncated else ""
                    md_lines.append(f'> [OCR]: "{excerpt}"{suffix}')
                elif status == "no_text_detected":
                    md_lines.append("> [OCR]: no text detected in this asset")
                elif status == "failed":
                    md_lines.append("> [OCR]: processing failed; text extraction is unverified")
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
                    "evidence": cat_item.attachment_evidence,
                    "text_extraction": cat_item.text_extraction or {"status": "not_requested"},
                    "sha256": cat_item.sha256,
                    "size_bytes": cat_item.size_bytes,
                    "local_rel_path": cat_item.local_rel_path,
                },
            }
            if cat_item.contact_info:
                json_entry["attachment"]["contact_info"] = cat_item.contact_info
            json_messages.append(json_entry)

        elif msg.is_omitted_media:
            md_lines.append(f"**[{msg.date_iso} {msg.time_str}] {msg.sender}:** [SOURCE MEDIA PLACEHOLDER] `{msg.content}`{edited_badge}")
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
            "evidence": c.attachment_evidence,
            "text_extraction": c.text_extraction or {"status": "not_requested"},
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
    paths = {"markdown": md_path, "json": json_path, "audit": audit_path}
    if ai_view:
        annotate_ai_payload(json_payload, messages, audit_payload, source_sha256)
        paths["ai_view"] = write_ai_markdown(json_payload, output_dir)
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    audit_path.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths
