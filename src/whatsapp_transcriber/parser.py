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
    attachment_type: Optional[str] = None  # "audio", "image", "video", "document", "sticker", "attachment"
    is_forwarded: bool = False
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

# Matches any media or document attachment reference in WhatsApp logs
ATTACHMENT_REGEX = re.compile(
    r"[\w\.\-]+\.(?:opus|ogg|m4a|mp3|wav|aac|flac|jpg|jpeg|png|webp|gif|mp4|3gp|mov|avi|mkv|pdf|docx?|xlsx?|csv|vcf|zip)",
    re.IGNORECASE
)

FORWARDED_INDICATORS = [
    "<mensagem encaminhada>",
    "mensagem encaminhada",
    "<forwarded message>",
    "forwarded message",
    "<reenviado>",
    "reenviado",
    "[mensagem encaminhada]",
    "[forwarded]"
]

def classify_attachment(filename: str) -> str:
    """Classifies attachment filename into a semantic category."""
    ext = Path(filename).suffix.lower()
    fname_upper = filename.upper()
    if ext in [".opus", ".ogg", ".m4a", ".mp3", ".wav", ".aac", ".flac"]:
        return "audio"
    if ext in [".jpg", ".jpeg", ".png", ".gif"]:
        return "image"
    if ext == ".webp" or fname_upper.startswith("STK-"):
        return "sticker"
    if ext in [".mp4", ".3gp", ".mov", ".avi", ".mkv"]:
        return "video"
    if ext in [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".vcf"]:
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

            att_match = ATTACHMENT_REGEX.search(content)
            attachment_file = att_match.group(0) if att_match else None
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
                is_forwarded=is_fwd
            )
            messages.append(current_msg)
        else:
            if current_msg:
                current_msg.content += "\n" + line
                if not current_msg.attachment_filename:
                    att_match = ATTACHMENT_REGEX.search(line)
                    if att_match:
                        current_msg.attachment_filename = att_match.group(0)
                        current_msg.attachment_type = classify_attachment(att_match.group(0))
                if not current_msg.is_forwarded:
                    if any(fwd in line.lower() for fwd in FORWARDED_INDICATORS):
                        current_msg.is_forwarded = True

    return messages
