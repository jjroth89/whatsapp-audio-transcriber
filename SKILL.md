---
name: whatsapp-audio-transcriber
description: "Automatically transcribes audio voice notes in exported WhatsApp chat archives (.zip or folder), aligns them chronologically, and embeds transcripts directly into conversation Markdown and JSON timelines."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, windows, macos, docker, container]
metadata:
  hermes:
    tags: [whatsapp, transcription, stt, whisper, faster-whisper, forensic, agentic, chat-analysis]
---

# WhatsApp Audio Transcriber Skill

A forensic and agentic skill that processes WhatsApp chat exports (ZIP or folder containing `_chat.txt`/`.md` and audio files), transcribes all voice notes, and embeds them directly into the conversation flow as chronological messages.

## Key Capabilities

1. **ZIP & Folder Ingestion:** Directly unzips and ingests WhatsApp chat archives without manual extraction.
2. **Conversation Anchoring:** Transcripts are inlined at the exact line in the conversation where the audio was sent or received.
3. **Shared / Forwarded Audio Discrepancy Detection:** Prevents timeline distortion when forwarded audios carry older filename timestamps. Identifies origin dates and tags them as `[FORWARDED/SHARED AUDIO]`.
4. **Adaptive STT Engine:**
   - **Local RTX 3090 / CUDA:** Uses `faster-whisper` (`large-v3`, FP16, VAD) on NVIDIA GPUs.
   - **Hermes Profile STT:** Automatically hooks into `HERMES_STT_ENDPOINT` when running inside a Hermes profile container.
5. **Dynamic Vocabulary Harvester:** Scans regular text messages in the chat to harvest acronyms, names, and technical terms to prime the STT model, merged with `vocab.txt`.
6. **Agent-First Deliverables:** Generates `enriched_chat.md` (for reading and LLM context prompts) and `enriched_chat.json` (structured message array).

## Usage for Hermes Agents

### 1. Basic Invocation (Automatic Engine Selection)

```bash
python -m whatsapp_transcriber.cli /path/to/exported_chat.zip
```

### 2. Specifying Output Directory and Custom Dictionary

```bash
python -m whatsapp_transcriber.cli /path/to/exported_chat.zip \
  --output /path/to/output_dir \
  --dict /path/to/vocab.txt \
  --language pt
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
