import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

@dataclass
class ChatMessage:
    line_number: int
    raw_timestamp: str
    date_iso: str
    time_str: str
    sender: str
    content: str
    is_system: bool = False
    attachment_filename: Optional[str] = None
    attachment_type: Optional[str] = None  # "audio", "image", "video", "document", "cad", "contact", "archive", "sticker", "attachment"
    is_forwarded: bool = False
    is_edited: bool = False
    is_omitted_media: bool = False
    parsed_datetime: Optional[datetime] = None

# Regex patterns for Android, iOS, and Markdown formats
ANDROID_PATTERN = re.compile(
    r"^(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\s*-\s*(.*?)$"
)

IOS_PATTERN = re.compile(
    r"^\[(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\]\s*(.*?)$"
)

MD_PATTERN = re.compile(
    r"^\*{0,2}\[(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4})\s+(\d{1,2}:\d{2}(?::\d{2})?)\]\s*([^:]+?)\*{0,2}:\s*(.*?)$"
)

KNOWN_EXTENSIONS = (
    "opus|ogg|m4a|mp3|wav|aac|flac|"
    "jpg|jpeg|png|webp|gif|heic|bmp|tiff|"
    "mp4|3gp|mov|avi|mkv|webm|"
    "pdf|docx?|xlsx?|pptx?|csv|txt|odt|ods|rtf|"
    "dwg|dxf|rvt|ifc|skp|step|stp|"
    "vcf|zip|rar|7z"
)

# Explicit WhatsApp attachment wrappers (preserves spaces, accents, and special chars)
ANDROID_ATTACH_REGEX = re.compile(
    r"^(.+?\.[A-Za-z0-9]{2,6})\s*\((?:file attached|arquivo anexado|ficheiro anexado)\)",
    re.IGNORECASE,
)
IOS_ATTACH_REGEX = re.compile(
    r"<(?:attached|anexado):\s*(.+?\.[A-Za-z0-9]{2,6})>",
    re.IGNORECASE,
)

# Fallback for plain/unwrapped attachment mentions
ATTACHMENT_REGEX = re.compile(
    rf"[\w\.\-\(\)\u00C0-\u017F]+\.(?:{KNOWN_EXTENSIONS})\b",
    re.IGNORECASE,
)

FORWARDED_INDICATORS = [
    "<mensagem encaminhada>",
    "mensagem encaminhada",
    "<forwarded message>",
    "forwarded message",
    "<reenviado>",
    "reenviado",
    "[mensagem encaminhada]",
    "[forwarded]",
]

OMITTED_MEDIA_INDICATORS = [
    "<media omitted>",
    "<mídia oculta>",
    "<midia oculta>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "document omitted",
    "sticker omitted",
    "imagem ocultada",
    "vídeo ocultado",
    "áudio ocultado",
    "documento ocultado",
]

EDITED_INDICATORS = [
    "<this message was edited>",
    "this message was edited",
    "<mensagem editada>",
    "esta mensagem foi editada",
]

def extract_attachment_filename(text: str) -> Optional[str]:
    """Extract attachment filename, preserving spaces/accents inside WhatsApp attachment markers."""
    clean = text.strip().strip("\u200e\u200f\ufeff")
    m = ANDROID_ATTACH_REGEX.search(clean)
    if m:
        return m.group(1).strip().strip("\u200e\u200f\ufeff")
    m = IOS_ATTACH_REGEX.search(clean)
    if m:
        return m.group(1).strip().strip("\u200e\u200f\ufeff")
    m = ATTACHMENT_REGEX.search(clean)
    if m:
        return m.group(0).strip()
    return None

def classify_attachment(filename: str) -> str:
    """Classifies attachment filename into a semantic category."""
    ext = Path(filename).suffix.lower()
    fname_upper = filename.upper()
    if ext in [".opus", ".ogg", ".m4a", ".mp3", ".wav", ".aac", ".flac"]:
        return "audio"
    if ext in [".jpg", ".jpeg", ".png", ".gif", ".heic", ".bmp", ".tiff"]:
        return "image"
    if ext == ".webp" or fname_upper.startswith("STK-"):
        return "sticker"
    if ext in [".mp4", ".3gp", ".mov", ".avi", ".mkv", ".webm"]:
        return "video"
    if ext in [".dwg", ".dxf", ".rvt", ".ifc", ".skp", ".step", ".stp"]:
        return "cad"
    if ext == ".vcf":
        return "contact"
    if ext in [".zip", ".rar", ".7z", ".tar", ".gz"]:
        return "archive"
    if ext in [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".pptx", ".ppt", ".txt", ".odt", ".ods", ".rtf"]:
        return "document"
    return "attachment"

def extract_chat_title(source_name: str) -> str:
    """Extracts a clean conversation title from WhatsApp file or zip name."""
    clean_stem = Path(source_name).stem
    m = re.search(r"(?:WhatsApp Chat with|Conversa do WhatsApp com)\s+(.+)$", clean_stem, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return clean_stem

def _parse_date(date_str: str) -> str:
    """Normalize date strings to ISO YYYY-MM-DD."""
    delimiters = ["/", "-", "."]
    delim = "/"
    for d in delimiters:
        if d in date_str:
            delim = d
            break
    parts = date_str.split(delim)
    if len(parts) == 3:
        p1, p2, p3 = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if len(p1) == 4:
            return f"{p1}-{int(p2):02d}-{int(p3):02d}"
        year = p3
        if len(year) == 2:
            year = "20" + year
        return f"{year}-{int(p2):02d}-{int(p1):02d}"
    return date_str

def parse_chat_log(text: str) -> List[ChatMessage]:
    """Parse WhatsApp conversation log into a structured list of ChatMessages."""
    lines = text.splitlines()
    messages: List[ChatMessage] = []
    current_msg: Optional[ChatMessage] = None

    for idx, line in enumerate(lines, 1):
        clean_line = line.strip("\u200e\u200f\ufeff")  # Strip unicode direction marks
        match = ANDROID_PATTERN.match(clean_line) or IOS_PATTERN.match(clean_line) or MD_PATTERN.match(clean_line)

        if match:
            date_raw, time_raw, remainder = match.groups()
            date_iso = _parse_date(date_raw)

            if ": " in remainder:
                sender_part, content_part = remainder.split(": ", 1)
                sender = sender_part.strip()
                content = content_part.strip()
                is_system = False
            else:
                sender = "System"
                content = remainder.strip()
                is_system = True

            content_lower = content.lower()
            is_fwd = any(fwd in content_lower for fwd in FORWARDED_INDICATORS)
            is_omitted = any(om in content_lower for om in OMITTED_MEDIA_INDICATORS)
            is_edited = any(ed in content_lower for ed in EDITED_INDICATORS)

            attachment_file = extract_attachment_filename(content)
            attachment_type = classify_attachment(attachment_file) if attachment_file else None

            current_msg = ChatMessage(
                line_number=idx,
                raw_timestamp=f"{date_raw} {time_raw}",
                date_iso=date_iso,
                time_str=time_raw.strip(),
                sender=sender,
                content=content,
                is_system=is_system,
                attachment_filename=attachment_file,
                attachment_type=attachment_type,
                is_forwarded=is_fwd,
                is_edited=is_edited,
                is_omitted_media=is_omitted,
            )
            messages.append(current_msg)
        else:
            if current_msg:
                current_msg.content += "\n" + line
                line_lower = line.lower()
                if not current_msg.attachment_filename:
                    att_file = extract_attachment_filename(line)
                    if att_file:
                        current_msg.attachment_filename = att_file
                        current_msg.attachment_type = classify_attachment(att_file)
                if not current_msg.is_forwarded:
                    if any(fwd in line_lower for fwd in FORWARDED_INDICATORS):
                        current_msg.is_forwarded = True
                if not current_msg.is_edited:
                    if any(ed in line_lower for ed in EDITED_INDICATORS):
                        current_msg.is_edited = True
                if not current_msg.is_omitted_media:
                    if any(om in line_lower for om in OMITTED_MEDIA_INDICATORS):
                        current_msg.is_omitted_media = True

    return messages
