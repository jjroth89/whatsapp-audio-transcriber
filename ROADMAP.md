# Roadmap & Backlog — `whatsapp-audio-transcriber`

This document tracks planned bug fixes, determinism improvements, and feature enhancements for upcoming releases.

---

## `v1.2.1` — Deterministic Corpus Hardening (implemented on feature branch) (`bug` / `enhancement`)

Target: eliminate downstream AI agent "missing media / incomplete verification" disclaimers when ingesting `*_agent_pack.zip`.

- [x] **Fix false-positive attachment matching (`parser.py`):**
  - Only classify a line as a file attachment if it carries an explicit WhatsApp attachment marker (`(file attached)`, `(arquivo anexado)`, `<attached: ...>`) or matches a real filename present in the source archive/directory.
  - Prevent plain-text mentions (e.g., `"send as .pdf"`, `"contract.docx"`) from being tagged as `missing_from_export`.
- [x] **Exclude raw `.txt` export from `*_agent_pack.zip` (`cli.py`):**
  - Keep `enriched_chat.md` and `enriched_chat.json` as the sole canonical conversation sources inside `--rezip` bundles so downstream agents do not cross-check raw `.opus` lines against `media/`.
- [x] **Reframe retention manifest as a Complete Text Corpus (`formatter.py`):**
  - Replace alarmist status wording (`purged`, `excluded`, `missing`) with positive, unambiguous corpus framing (`complete_text_corpus`, `text_equivalent_verified`, `decorative_stickers_skipped`).
  - Add an explicit ingestion contract header instructing downstream LLMs not to flag transcribed A/V streams or skipped stickers as missing data.
- [x] **Add Top-of-File Corpus Index (`formatter.py`):**
  - Generate a compact inventory table at the top of `enriched_chat.md` listing all retained PDFs, CAD drawings, `.vcf` contacts, and images with their timestamps and relative links.

---

## `v1.3.0` — Local Deterministic Document & Photo Text Extraction (`enhancement`)

Target: pre-extract text from PDFs and images locally before cloud LLM ingestion, minimizing reliance on downstream agent OCR/reading.

- [ ] **Native PDF Text Extraction (`pymupdf` / `fitz`):**
  - Extract embedded text deterministically from PDF attachments in `media/`.
  - Store extracted text in `enriched_chat.json` (and reference/inline cleanly in `enriched_chat.md`).
  - Automatically detect scanned/image-only PDFs (0 extracted characters) and route them to OCR.
- [ ] **Local Photo & Scanned-PDF OCR (`rapidocr-onnxruntime` via optional `[ocr]` extra / `--ocr` flag):**
  - Pure-`pip` ONNX OCR with zero external binary installers required on Windows/Linux/macOS.
  - Extract text from `.jpg`/`.png` photos (receipts, bank slips, screenshots, handwritten/printed notes) and scanned PDF pages.
  - Inline `> [OCR]: "..."` under `[IMAGE]` / `[DOCUMENT]` entries; cleanly tag text-free visual photos.

### v1.2.1 verification note

Synthetic coverage includes filename-prose false positives, explicit missing references, duplicate flattened basenames, agent ZIP allowlisting with legitimate text documents, folder-source preservation, stale-output rejection and truthful schema-v2 coverage. Local runner acceptance is tracked separately; unchecked v1.3.0 items remain pending.
