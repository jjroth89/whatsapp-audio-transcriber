"""Optional local PDF and image text extraction.

No cloud OCR, Tesseract executable, or mandatory OCR imports. OCR results
describe the source attachment; packaging the original is a separate choice.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

MAX_DOCUMENT_BYTES = 30 * 1024 * 1024
MAX_PAGES = 200
MAX_TEXT_CHARS = 100_000
PDF_RENDER_SCALE = 1.5


def create_ocr_engine():
    """Initialize RapidOCR 1.4.x only when the user selects --ocr."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "Optional OCR dependencies are missing; install whatsapp-audio-transcriber[ocr]"
        ) from exc
    return RapidOCR()


def _recognize(image: Any, engine: Any) -> Dict[str, Any]:
    """RapidOCR 1.4.x returns (lines, elapsed); sort by visible reading position."""
    result, _elapsed = engine(image)
    if not result:
        return {"text": "", "confidence": None}
    lines = []
    for item in result:
        if len(item) < 3 or not str(item[1]).strip():
            continue
        box, text, score = item[0], str(item[1]).strip(), item[2]
        try:
            top = min(float(pt[1]) for pt in box)
            left = min(float(pt[0]) for pt in box)
        except (TypeError, ValueError, IndexError):
            top, left = float("inf"), float("inf")
        lines.append((round(top, 1), round(left, 1), text, float(score)))
    lines.sort(key=lambda row: (row[0], row[1], row[2]))
    return {
        "text": "\n".join(item[2] for item in lines),
        "confidence": round(sum(item[3] for item in lines) / len(lines), 4) if lines else None,
    }


def _result(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = "\n\n".join(page.get("text", "") for page in pages if page.get("text")).strip()
    truncated = len(text) > MAX_TEXT_CHARS
    if truncated:
        text = text[:MAX_TEXT_CHARS]
    failed = any(page.get("status") == "failed" for page in pages)
    methods = sorted({page["method"] for page in pages if page.get("text")})
    return {
        "status": "failed" if failed else ("extracted" if text else "no_text_detected"),
        "method": "_and_".join(methods) if methods else "none",
        "text": text,
        "pages": pages,
        "truncated": truncated,
    }


def extract_image_text(path: Path, engine: Any) -> Dict[str, Any]:
    """Read a local JPEG/PNG; OCR errors are explicit and do not fabricate text."""
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        return {"status": "failed", "method": "rapidocr", "text": "",
                "pages": [], "error": "image_size_limit"}
    try:
        item = _recognize(str(path), engine)
        return _result([{"page": 1, "method": "rapidocr", **item,
                         "status": "extracted" if item["text"] else "no_text_detected"}])
    except Exception as exc:
        return {"status": "failed", "method": "rapidocr", "text": "", "pages": [],
                "error": type(exc).__name__}


def extract_pdf_text(path: Path, engine: Any) -> Dict[str, Any]:
    """Extract PDF text with pypdf; render text-free pages via pypdfium2 + RapidOCR."""
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        return {"status": "failed", "method": "pdf_native", "text": "",
                "pages": [], "error": "pdf_size_limit"}
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path), strict=False)
        if len(reader.pages) > MAX_PAGES:
            return {"status": "failed", "method": "pdf_native", "text": "",
                    "pages": [], "error": "pdf_page_limit"}
        result_pages: List[Dict[str, Any]] = []
        pdfium_doc: Optional[Any] = None
        for index, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                result_pages.append({"page": index + 1, "method": "pdf_native",
                                     "text": text, "status": "extracted"})
                continue
            try:
                if pdfium_doc is None:
                    import pypdfium2 as pdfium
                    import numpy as np
                    pdfium_doc = pdfium.PdfDocument(str(path))
                page_bitmap = pdfium_doc[index].render(scale=PDF_RENDER_SCALE)
                image = page_bitmap.to_pil().convert("RGB")
                # RapidOCR's ndarray input uses OpenCV BGR channel order.
                import numpy as np
                bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
                item = _recognize(bgr, engine)
                result_pages.append({"page": index + 1, "method": "rapidocr", **item,
                                     "status": "extracted" if item["text"] else "no_text_detected"})
            except Exception as exc:
                result_pages.append({"page": index + 1, "method": "rapidocr",
                                     "text": "", "status": "failed",
                                     "error": type(exc).__name__})
        if pdfium_doc is not None:
            pdfium_doc.close()
        return _result(result_pages)
    except Exception as exc:
        return {"status": "failed", "method": "pdf_native", "text": "",
                "pages": [], "error": type(exc).__name__}


def extract_media_text(path: Path, engine: Any) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path, engine)
    if suffix in {".jpg", ".jpeg", ".png"}:
        return extract_image_text(path, engine)
    return {"status": "not_requested", "method": "none", "text": "", "pages": []}
