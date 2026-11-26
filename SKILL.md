---
name: whatsapp-audio-transcriber
description: "Automatically transcribes audio voice notes in exported WhatsApp chat archives (.zip or folder), aligns them chronologically, semantically tags documents/media, and embeds transcripts directly into conversation Markdown and JSON timelines."
version: 1.3.1
author: Hermes Agent
license: MIT
platforms: [linux, windows, macos, docker, container]
metadata:
  hermes:
    tags: [whatsapp, transcription, stt, whisper, faster-whisper, forensic, agentic, chat-analysis, multimodal]
---

# WhatsApp Audio Transcriber Skill

A forensic and agentic skill that processes WhatsApp chat exports (ZIP or folder containing `_chat.txt`/`.md` and audio files), transcribes all voice notes, and embeds them directly into the conversation flow as chronological messages.

## Key Capabilities

1. **ZIP & Folder Ingestion with Unified `--media` Retention Modes:**
   - **`--media audio` (Default):** Extracts only audio voice notes and chat logs.
   - **`--media all`:** Unpacks complete media archive (photos, videos, documents, CAD, contacts, audio).
   - **`--media no-videos`:** Unpacks everything except heavy video files.
   - **`--media docs`:** Preserves PDFs, documents, images, CAD drawings, and contacts while purging raw audio/video binaries after transcription.
   - **`--media none`:** Lean mode — purges all raw media after transcription, leaving only compact text/JSON artifacts.
   - **`--transcribe-videos`:** Extracts & transcribes spoken audio from video files via PyAV (even if video binaries are purged).
   - **`--rezip`:** Compresses final deliverables into a ready-to-share `<slug>_agent_pack.zip` bundle for cloud agents or ChatGPT Desktop.
2. **Complete Text Corpus Contract (schema v2):** Distinguishes verified audio text equivalents, retained originals, source-unavailable references and policy packaging without claiming that unread PDFs/images were interpreted.\n3. **Semantic Tagging, vCard Inlining & Timestamped Docs:** Classifies `[CAD]`, `[CONTACT]` (inline parsed `.vcf` details), `[DOCUMENT]`, `[IMAGE]`, `[VIDEO]`, `[STICKER]` and prefixes non-timestamped docs/CAD files with `YYYY-MM-DD_HHMM_` in `media/` with URL-encoded Markdown links.
3. **Conversation Anchoring:** Transcripts are inlined at the exact line in the conversation where the audio was sent or received.
4. **Shared / Forwarded Audio Discrepancy Detection:** Prevents timeline distortion when forwarded audios carry older filename timestamps. Identifies origin dates and tags them as `[FORWARDED/SHARED AUDIO]`.
5. **Optional Local OCR (`--ocr`):** PDF embedded text with pypdf; scanned PDF pages rendered with pypdfium2 and local RapidOCR 1.4.x; JPEG/PNG text captured without cloud OCR or Tesseract binaries. OCR results are attached to `enriched_chat.json` and excerpted in Markdown.
6. **Adaptive Hardware & Cache Engine:** Automatically runs in CUDA FP16 mode on NVIDIA GPUs (e.g., RTX 3090) with zero setup, with graceful 8-bit quantized CPU fallback for portable systems. Reuses existing model caches without re-downloading.
7. **Agent-First Deliverables:** Generates `enriched_chat.md` (for reading and LLM context prompts) and `enriched_chat.json` (structured message array).

## Usage for Hermes Agents

### 1. Basic Invocation (Standard Extraction + Rezip)

```bash
whatsapp-transcribe /path/to/exported_chat.zip --rezip
```

### 2. Docs, Images, CAD & Contacts Only (With Video Transcripts, Purged Raw A/V)

```bash
whatsapp-transcribe /path/to/exported_chat.zip --media docs --transcribe-videos --rezip
```

### 3. Full Multimodal Extraction (For Vision & Document Agents)

```bash
whatsapp-transcribe /path/to/exported_chat.zip --media all --transcribe-videos --rezip
```

### 4. Lean Mode (Max Disk Savings, Artifacts Only)

```bash
whatsapp-transcribe /path/to/exported_chat.zip --media none --rezip
```

### 5. Optional Local PDF/Photo OCR

Install `pip install -e '.[ocr]'` and add `--ocr`. With `--media none`, originals are processed temporarily but omitted from the pack; extracted text remains in JSON and Markdown. OCR status is `not_requested`, `extracted`, `no_text_detected`, or `failed`. Text results do not imply visual comprehension of pictures or CAD.

```bash
whatsapp-transcribe chat.zip --media docs --ocr --rezip
```

### 6. Explicit Engine & Model Cache Selection

* **Local NVIDIA GPU (RTX 3090 with cached weights):**
  ```bash
  whatsapp-transcribe chat.zip --engine local --device cuda --model large-v3 --compute-type float16 --model-dir /path/to/weights
  ```

* **Hermes Profile STT Service:**
  ```bash
  whatsapp-transcribe chat.zip --engine hermes --stt-url "http://localhost:8080/v1/audio/transcriptions"
  ```

## Flag Reference

| Flag | Purpose | Default |
| :--- | :--- | :--- |
| `-o`, `--output` | Destination folder | `<slug>_transcribed` |
| `--media` | Retention policy (`audio`, `all`, `no-videos`, `docs`, `none`) | `audio` |
| `--transcribe-videos` | Extract & transcribe audio tracks from videos via PyAV | `False` |
| `--rezip` | Package output directory into `<slug>_agent_pack.zip` | `False` |
| `--ocr` | Optional native PDF text and local RapidOCR | `False` |
| `-e`, `--engine` | STT engine (`auto`, `local`, `hermes`) | `auto` |
| `-m`, `--model` | Whisper model (`large-v3`, `large-v3-turbo`, `base`, etc.) | `large-v3` |
| `-d`, `--device` | Device (`auto`, `cuda`, `cpu`) | `auto` |
| `-c`, `--compute-type` | Precision (`auto`, `float16`, `int8`, `float32`) | `auto` |
| `--model-dir` | Path to load/cache local Whisper weights | `None` (HuggingFace cache) |
| `--dict` | Custom vocabulary / specialized terms file (`vocab.txt`) | `vocab.txt` |
| `-q`, `--quiet` | Suppress console progress bars and status logs | `False` |

## Output Artifacts

Each execution outputs:
- `enriched_chat.md`: Full human- and LLM-readable conversation transcript with inlined voice notes and relative links to attachments.
- `enriched_chat.json`: Structured array of message objects with audio metadata, duration, confidence, and timestamps.
- `audio_audit.json`: SHA-256 hashes, durations, alerts, `media_catalog`, and processing metrics.
- `media/`: Preserved non-video or multimodal media assets (PDFs, images, stickers, audios) depending on the chosen extraction mode.
- `<slug>_agent_pack.zip`: Self-contained compressed bundle ready for immediate ingestion by ChatGPT Desktop, Claude, or autonomous multi-agent workflows (created when `--rezip` is passed).
