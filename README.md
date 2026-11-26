# WhatsApp Audio Transcriber

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA%2012%20%2F%20RTX%203090-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![faster-whisper](https://img.shields.io/badge/engine-faster--whisper%20large--v3-orange.svg)](https://github.com/SYSTRAN/faster-whisper)
[![Hermes Agent](https://img.shields.io/badge/skill-Hermes%20Agent%20Ready-purple.svg)](https://github.com/jjroth89)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An agentic and forensic audio transcription pipeline for **WhatsApp chat exports**. It ingests standard WhatsApp `.zip` packages or folders, transcribes all audio voice notes with high fidelity, and embeds them directly into the chronological conversation timeline.

Designed for AI agent workflows (**Hermes Agent**, **ChatGPT Desktop**, **Claude**, **NotebookLM**) and legal/contractual analysis where audio provenance, exact timestamps, and conversational context are critical.

---

## 🌟 Key Features

* **📦 Unified Media Retention (`--media`):** Choose `audio` (default), `all`, `no-videos`, `docs` (keeps PDFs, images, CAD & contacts while omitting original A/V), or `none` (lean artifacts only).
* **🎬 Video Audio Transcription (`--transcribe-videos`):** Extracts and transcribes spoken audio from `.mp4`, `.mov`, `.3gp`, and `.mkv` via PyAV (no external `ffmpeg` required)—even when video binaries are excluded from the final pack.
* **🛡️ Explicit Retention & Provenance Manifest:** Records evidence-backed source/retention categories and truthful `complete_text_corpus` status (`schema_version: 2`); unverified sources remain visibly partial.
* **🕒 Chronological Document Timestamping:** Automatically prefixes non-timestamped files (`.pdf`, `.dwg`, `.vcf`) with `YYYY-MM-DD_HHMM_` in `media/` with URL-encoded Markdown links.
* **📇 Native vCard & CAD Support:** Parses `.vcf` contacts inline (`[CONTACT | Name | Phone | Org]`) and classifies AutoCAD/engineering drawings (`[CAD]`: `.dwg`, `.dxf`, `.rvt`, `.ifc`).
* **🗜️ Instant Agent Packaging (`--rezip`):** Packages the canonical enriched artifacts plus retained `media/` into a standalone `<slug>_agent_pack.zip` ready for ChatGPT Desktop or cloud agents.
* **🎯 Line-Anchored Conversational Placement:** Transcripts are inlined directly at the exact line where the voice note was sent or received.
* **🛡️ Shared & Forwarded Audio Provenance Detection:** Detects whether an audio note was recorded live in the chat (`PTT-`) or forwarded from another conversation (`AUD-` or date discrepancy). Never misplaces forwarded audios in the timeline!
* **⚡ High-Performance GPU Inference:** Optimized for **NVIDIA GeForce RTX 3090** (and CUDA 12) using `faster-whisper` (`large-v3` in FP16), with Silero VAD filtering and word-level timestamps.
* **🤖 Dual Engine (Local GPU + Hermes Profile):** Automatically detects whether it is running on a bare-metal CUDA workstation or inside a Hermes container profile with a centralized STT service.
* **🧠 Dynamic Vocabulary Harvester:** Analyzes text messages in the chat to automatically extract proper nouns, acronyms (e.g. *CNO, ITBI, Sero*), and jargon to prime the transcription prompt, merged with a customizable `vocab.txt`.
* **📄 Agent-First Deliverables:** Generates both `enriched_chat.md` (clean markdown for reading and LLMs) and `enriched_chat.json` (structured message schema).

---

## 📂 Architecture

```
Exported WhatsApp ZIP
        │
        ├──> Extract & Parse Chat Log (iOS / Android / MD)
        │           │
        │           ├──> Scan Text -> Dynamic Vocabulary Harvester (+ vocab.txt)
        │           │
        │           └──> Match Audio References (PTT vs AUD, Date Check)
        │                       │
        │                       ▼
        │              STT Engine Selector
        │           ┌───────────┴───────────┐
        │           ▼                       ▼
        │    Local GPU (RTX 3090)    Hermes Profile STT
        │    faster-whisper FP16     HTTP API / whisper-1
        │           │                       │
        │           └───────────┬───────────┘
        │                       │
        ▼                       ▼
Enriched Output:
  ├── enriched_chat.md       (Timeline with inlined audio notes)
  ├── enriched_chat.json     (Structured message objects)
  ├── audio_audit.json       (Hashes, durations, confidence, media catalog)
  ├── media/                 (Extracted attachments depending on mode)
  └── [chat]_agent_pack.zip  (Standalone bundle when using --rezip)
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/jjroth89/whatsapp-audio-transcriber.git
cd whatsapp-audio-transcriber

# Create environment (e.g., Python 3.10, 3.11, or 3.12)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and CLI
pip install -e .

# For NVIDIA GPU acceleration (CUDA 12):
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12
```

### 2. Usage Examples

**Standard Mode (audio & chat only, agent pack):**
```bash
whatsapp-transcribe /path/to/WhatsApp_Chat.zip --rezip
```

**Docs + Images + CAD + Contacts Only (transcribes audio & videos, purges raw A/V binaries):**
```bash
whatsapp-transcribe /path/to/WhatsApp_Chat.zip --media docs --transcribe-videos --rezip
```

**Full Multimodal Extraction (documents, images, CAD, contacts, audios, and videos):**
```bash
whatsapp-transcribe /path/to/WhatsApp_Chat.zip --media all --transcribe-videos --rezip
```

**Lean Mode (artifacts only, purges all raw media post-transcription):**
```bash
whatsapp-transcribe /path/to/WhatsApp_Chat.zip --media none --rezip
```

---

## ⚙️ Command-Line Options

```text
usage: whatsapp-transcribe [-h] [-o OUTPUT] [-l LANGUAGE]
                           [--media {audio,all,no-videos,docs,none}]
                           [--transcribe-videos] [--rezip]
                           [-e {auto,local,hermes}] [-m MODEL]
                           [-d {auto,cuda,cpu}] [-c COMPUTE_TYPE]
                           [--model-dir MODEL_DIR] [--dict DICT]
                           [-b BATCH_SIZE] [--stt-url STT_URL]
                           [--api-key API_KEY] [-q]
                           input_path

Core:
  input_path             Path to exported WhatsApp .zip file or folder.
  -o, --output DIR       Destination folder (default: <input_name>_transcribed).
  -l, --language CODE    Spoken language code (default: pt).

Media & Packaging:
  --media MODE           Retention policy: audio (default), all, no-videos, docs, none.
  --transcribe-videos    Extract & transcribe audio tracks from video files.
  --rezip                Bundle final output directory into <output>_agent_pack.zip.

Engine & Model:
  -e, --engine CHOICE    STT engine: auto, local, or hermes (default: auto).
  -m, --model MODEL      Whisper model name (default: large-v3).
  -d, --device DEVICE    Hardware device: auto, cuda, or cpu (default: auto).
  -c, --compute-type PRE Precision: auto, float16, int8_float16, int8, float32.
  --model-dir DIR        Custom folder for loading/caching Whisper weights.
  --dict FILE            Custom vocabulary file (vocab.txt).

Advanced / Remote STT:
  -b, --batch-size NUM   Inference batch size (default: 8).
  --stt-url URL          Custom endpoint URL for Hermes STT profile.
  --api-key KEY          API key for Hermes or OpenAI STT service.
  -q, --quiet            Suppress progress output.
```

---

## 📝 Example Output (`enriched_chat.md`)

```markdown
# Project Coordination Chat — Enriched Transcript

**[2024-01-15 12:57] Alice:** [AUDIO NOTE | Duration: 24.5s | File: `PTT-20240115-WA0001.opus`]
> "Good afternoon Bob, I reviewed the proposal with our technical lead and we agree on the scope..."

**[2024-01-15 14:55] Bob:** Sounds good to me, let's proceed with the draft agreement.

**[2025-10-23 18:19] Charlie:** [AUDIO NOTE | Duration: 45.2s | File: `PTT-20251023-WA0002.opus`]
> "Hi team, I confirmed the delivery timeline for phase two. We are on schedule..."

**[2025-10-24 10:15] Bob:** [FORWARDED/SHARED AUDIO | Originated: 2024-03-12 | Duration: 30.1s | File: `AUD-20240312-WA0003.opus`]
> "Here is the architectural audio note forwarded from the external engineering team..."
```

---

## 🤖 AI Agent & ChatGPT Desktop Guide

For complete instructions on integrating this tool with autonomous agents and desktop LLMs, see **[`AGENTS.md`](AGENTS.md)**.

* **ChatGPT Desktop:** Run transcription locally -> drag-and-drop `enriched_chat.md` into ChatGPT Desktop -> use provided prompt templates for forensic analysis.
* **Hermes Agent:** Native skill execution via [`SKILL.md`](SKILL.md) with automatic STT profile endpoint detection and structured JSON output.

---

## 🧩 Hermes Agent Skill Integration

This repository includes a native [`SKILL.md`](SKILL.md) conforming to Hermes Agent standards.
To use it inside Hermes:

1. Copy or symlink this folder to your Hermes skills directory.
2. The agent can invoke the tool directly using standard bash execution or python module calls.
3. If running inside a container with `HERMES_STT_ENDPOINT` set, the tool delegates transcription to the container profile automatically.

---

## 📄 License

Released under the [MIT License](LICENSE).
Created by [jjroth89](https://github.com/jjroth89).

## v1.2.1 — Deterministic Corpus Contract

The exported raw conversation file is used as input but is **not** included in `*_agent_pack.zip`. The package contains `enriched_chat.md`, `enriched_chat.json`, `audio_audit.json`, and selected `media/` assets (including legitimate `.txt` documents). Source input files are never modified. When using a directory input, select an output directory outside that source.

Attachment classification requires either an explicit WhatsApp marker or an exact filename present in the source inventory; a plain mention of `contract.docx` without such evidence remains normal text. Duplicate basenames and nonempty output directories are rejected instead of silently losing or repackaging data.

The `package_policy.schema_version=2` / `complete_text_corpus.status` describes what could actually be verified **within the supplied export**, never completeness of all real-world communications or interpretation of document images. A partial result is evidence of a genuine unverified media reference, not a missing file inferred from arbitrary prose. The Markdown starts with a Document, Contact & Photo Index of retained, linked files.
