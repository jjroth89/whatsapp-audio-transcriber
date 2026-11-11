---
name: whatsapp-audio-transcriber
description: "Automatically transcribes audio voice notes in exported WhatsApp chat archives (.zip or folder), aligns them chronologically, semantically tags documents/media, and embeds transcripts directly into conversation Markdown and JSON timelines."
version: 1.1.0
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

1. **ZIP & Folder Ingestion with Extraction Modes:**
   - **Standard (Default):** Selectively extracts only audio voice notes and chat logs, skipping bulky videos/images.
   - **`--extract-all`:** Unpacks complete media archive (photos, videos, documents) for full multimodal agent ingestion.
   - **`--no-videos`:** Unpacks everything except heavy video files.
   - **`--lean`:** Purges raw audio files after transcription, leaving only compact text/JSON artifacts.
   - **`--rezip`:** Compresses final deliverables into a ready-to-share bundle for cloud agents or ChatGPT Desktop.
2. **Semantic Attachment Tagging & Catalog:** Classifies non-audio files (`[DOCUMENT]`, `[IMAGE]`, `[VIDEO]`, `[STICKER]`) and formats them as relative links. Generates a structured `media_catalog` in the audit JSON.
3. **Conversation Anchoring:** Transcripts are inlined at the exact line in the conversation where the audio was sent or received.
4. **Shared / Forwarded Audio Discrepancy Detection:** Prevents timeline distortion when forwarded audios carry older filename timestamps. Identifies origin dates and tags them as `[FORWARDED/SHARED AUDIO]`.
5. **Adaptive Hardware & Cache Engine:** Automatically runs in CUDA FP16 mode on NVIDIA GPUs (e.g., RTX 3090) with zero setup, with graceful 8-bit quantized CPU fallback for portable systems. Reuses existing model caches without re-downloading.
6. **Agent-First Deliverables:** Generates `enriched_chat.md` (for reading and LLM context prompts) and `enriched_chat.json` (structured message array).

## Usage for Hermes Agents

### 1. Basic Invocation (Standard Extraction + Rezip)

```bash
python -m whatsapp_transcriber.cli /path/to/exported_chat.zip --rezip
```

### 2. Full Multimodal Extraction (For Vision & Document Agents)

```bash
python -m whatsapp_transcriber.cli /path/to/exported_chat.zip --extract-all --rezip
```

### 3. Lean Mode (Max Disk Savings, Artifacts Only)

```bash
python -m whatsapp_transcriber.cli /path/to/exported_chat.zip --lean --rezip
```

### 3. Explicit Engine Selection

* **Local NVIDIA GPU (RTX 3090):**
  ```bash
  python -m whatsapp_transcriber.cli chat.zip --engine local --device cuda --model large-v3 --compute-type float16
  ```

* **Hermes Profile STT Service:**
  ```bash
  python -m whatsapp_transcriber.cli chat.zip --engine hermes --stt-url "http://localhost:8080/v1/audio/transcriptions"
  ```

## Output Artifacts

Each execution outputs:
- `enriched_chat.md`: Full human- and LLM-readable conversation transcript with inlined voice notes.
- `enriched_chat.json`: Structured array of message objects with audio metadata, duration, confidence, and timestamps.
- `audio_audit.json`: SHA-256 hashes, durations, alerts, and processing metrics.
- `<output_dir>.zip`: Compressed bundle ready for immediate ingestion by ChatGPT Desktop, Claude, or legal analysis workflows.
