import re
from dataclasses import dataclass, field
from datetime import datetime
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
    is_forwarded: bool = False
    parsed_datetime: Optional[datetime] = None

# Regex patterns for Android and iOS formats
# Android: "12/06/2024 12:57 - Sender: Content" or "12/06/2024, 12:57 - Sender: Content"
ANDROID_PATTERN = re.compile(
    r"^(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\s*-\s*(.*?)$"
)

# iOS: "[12/06/2024, 12:57:15] Sender: Content"
IOS_PATTERN = re.compile(
    r"^\[(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\]\s*(.*?)$"
)

# Markdown style: "**[12/06/2024 12:57] Sender:** Content"
MD_PATTERN = re.compile(
    r"^\*{0,2}\[(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4})\s+(\d{1,2}:\d{2}(?::\d{2})?)\]\s*([^:]+?)\*{0,2}:\s*(.*?)$"
)

# Common attachment indicators in PT, EN, ES
AUDIO_EXT_REGEX = re.compile(r"[\w\-]+\.(?:opus|ogg|m4a|mp3|wav|aac|flac)", re.IGNORECASE)
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
        # If year is first (YYYY-MM-DD)
        if len(p1) == 4:
            return f"{p1}-{int(p2):02d}-{int(p3):02d}"
        # If year is last (DD/MM/YYYY or MM/DD/YYYY)
        year = p3
        if len(year) == 2:
            year = "20" + year
        # Standard WhatsApp export in BR/EU is DD/MM/YYYY
        return f"{year}-{int(p2):02d}-{int(p1):02d}"
    return date_str

def parse_chat_log(text: str) -> List[ChatMessage]:
    """Parse WhatsApp conversation log into a structured list of ChatMessages."""
    lines = text.splitlines()
    messages: List[ChatMessage] = []
    current_msg: Optional[ChatMessage] = None

    for idx, line in enumerate(lines, 1):
        clean_line = line.strip("\u200e\u200f\ufeff")  # Strip invisible unicode RTL/LTR marks
        match = ANDROID_PATTERN.match(clean_line) or IOS_PATTERN.match(clean_line) or MD_PATTERN.match(clean_line)

        if match:
            date_raw, time_raw, remainder = match.groups()
            date_iso = _parse_date(date_raw)

            # Check if this message has a sender or is a system notification
            if ": " in remainder:
                sender_part, content_part = remainder.split(": ", 1)
                sender = sender_part.strip()
                content = content_part.strip()
                is_system = False
            else:
                sender = "System"
                content = remainder.strip()
                is_system = True

            # Detect forwarded markers
            content_lower = content.lower()
            is_fwd = any(fwd in content_lower for fwd in FORWARDED_INDICATORS)

            # Detect audio filename reference
            audio_match = AUDIO_EXT_REGEX.search(content)
            attachment_file = audio_match.group(0) if audio_match else None

            current_msg = ChatMessage(
                line_number=idx,
                raw_timestamp=f"{date_raw} {time_raw}",
                date_iso=date_iso,
                time_str=time_raw.strip(),
                sender=sender,
                content=content,
                is_system=is_system,
                attachment_filename=attachment_file,
                is_forwarded=is_fwd
            )
            messages.append(current_msg)
        else:
            # Multi-line message continuation
            if current_msg:
                current_msg.content += "\n" + line
                # Check for audio attachment in subsequent lines
                if not current_msg.attachment_filename:
                    audio_match = AUDIO_EXT_REGEX.search(line)
                    if audio_match:
                        current_msg.attachment_filename = audio_match.group(0)
                if not current_msg.is_forwarded:
                    if any(fwd in line.lower() for fwd in FORWARDED_INDICATORS):
                        current_msg.is_forwarded = True

    return messages
