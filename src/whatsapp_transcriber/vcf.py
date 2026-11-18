import quopri
import re
from pathlib import Path
from typing import Any, Dict, List


def _unfold_vcard_lines(raw_text: str) -> List[str]:
    """Unfold RFC 2425/6350 folded lines and quoted-printable soft line breaks."""
    raw_lines = raw_text.splitlines()
    unfolded: List[str] = []
    for line in raw_lines:
        if line.startswith((" ", "\t")) and unfolded:
            unfolded[-1] += line[1:]
        elif unfolded and unfolded[-1].endswith("=") and "QUOTED-PRINTABLE" in unfolded[-1].upper():
            unfolded[-1] = unfolded[-1][:-1] + line
        else:
            unfolded.append(line)
    return unfolded


def _decode_value(value: str, params: str) -> str:
    """Decode QUOTED-PRINTABLE UTF-8 vCard values if flagged in parameters."""
    if "QUOTED-PRINTABLE" in params.upper():
        try:
            decoded_bytes = quopri.decodestring(value.encode("latin-1", errors="ignore"))
            return decoded_bytes.decode("utf-8", errors="replace").strip()
        except Exception:
            pass
    return value.replace("\\;", ";").replace("\\,", ",").replace("\\n", " ").strip()


def parse_vcf_file(path: Path) -> Dict[str, Any]:
    """
    Parse a .vcf (vCard) contact file into structured contact metadata.
    Supports vCard 2.1, 3.0, and 4.0 exports from Android and iOS WhatsApp.
    """
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    lines = _unfold_vcard_lines(raw_text)

    fn_name = ""
    n_name = ""
    phones: List[str] = []
    emails: List[str] = []
    org = ""
    title = ""

    for line in lines:
        if ":" not in line:
            continue
        prop_part, val_part = line.split(":", 1)
        prop_upper = prop_part.upper()
        prop_base = prop_upper.split(";", 1)[0].split(".", 1)[-1]
        decoded = _decode_value(val_part, prop_part)

        if not decoded:
            continue

        if prop_base == "FN" and not fn_name:
            fn_name = decoded
        elif prop_base == "N" and not n_name:
            parts = [p.strip() for p in decoded.split(";") if p.strip()]
            # N format is Family;Given;Middle;Prefix;Suffix
            if len(parts) >= 2:
                n_name = f"{parts[1]} {parts[0]}".strip()
            elif parts:
                n_name = parts[0]
        elif prop_base == "TEL":
            clean_phone = re.sub(r"\s+", " ", decoded).strip()
            if clean_phone and clean_phone not in phones:
                phones.append(clean_phone)
        elif prop_base == "EMAIL":
            if decoded not in emails:
                emails.append(decoded)
        elif prop_base == "ORG" and not org:
            org = decoded.replace(";", " - ").strip(" -")
        elif prop_base == "TITLE" and not title:
            title = decoded

    display_name = fn_name or n_name or path.stem
    return {
        "name": display_name,
        "phones": phones,
        "emails": emails,
        "organization": org,
        "title": title,
    }


def format_vcf_summary(contact: Dict[str, Any]) -> str:
    """Format parsed vCard dictionary into a compact single-line summary."""
    parts = [contact.get("name", "Unknown Contact")]
    if contact.get("organization"):
        parts.append(f"Org: {contact['organization']}")
    if contact.get("phones"):
        parts.append(f"Tel: {', '.join(contact['phones'])}")
    if contact.get("emails"):
        parts.append(f"Email: {', '.join(contact['emails'])}")
    return " | ".join(parts)
