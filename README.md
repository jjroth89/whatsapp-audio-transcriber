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

* **📦 Direct ZIP Ingestion:** Seamlessly unzips and parses WhatsApp exports containing `_chat.txt` (iOS), `Conversa do WhatsApp.txt` (Android), or Markdown chat logs alongside media files.
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
  ├── enriched_chat.md   (Timeline with inlined audio notes)
  ├── enriched_chat.json (Structured message objects)
  ├── audio_audit.json   (Hashes, durations, confidence)
  └── [chat]_transcribed.zip
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/jjroth89/whatsapp-audio-transcriber.git
cd whatsapp-audio-transcriber

# Create environment (e.g., Python 3.11 or 3.12)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For NVIDIA GPU acceleration (CUDA 12):
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12
```

### 2. Basic Usage

Transcribe an exported WhatsApp chat ZIP:

```bash
python -m whatsapp_transcriber.cli /path/to/WhatsApp_Chat.zip
```

Output files will be generated in `WhatsApp_Chat_transcribed/`.

---

## ⚙️ Command-Line Options

```text
usage: whatsapp-transcribe [-h] [-o OUTPUT] [-e {auto,local,hermes}] [-m MODEL]
                           [-d {cuda,cpu}] [-c COMPUTE_TYPE] [-b BATCH_SIZE]
                           [-l LANGUAGE] [--dict DICT] [--stt-url STT_URL]
                           [--api-key API_KEY] [--zip-output] [--quiet]
                           input_path

Positional Arguments:
  input_path             Path to exported WhatsApp .zip file or folder.

Options:
  -o, --output DIR       Destination folder (default: <input_name>_transcribed)
  -e, --engine CHOICE    STT engine: auto, local, or hermes (default: auto)
  -m, --model MODEL      Whisper model name (default: large-v3)
  -d, --device DEVICE    cuda or cpu (default: cuda)
  -c, --compute-type PRE float16, int8_float16, float32 (default: float16)
  -b, --batch-size NUM   Inference batch size (default: 8)
  -l, --language CODE    Spoken language code (default: pt)
  --dict FILE            Custom dictionary file for uncommon terms (vocab.txt)
  --stt-url URL          Endpoint URL for Hermes Profile or OpenAI STT
  --api-key KEY          API key for Hermes or OpenAI STT service
  --zip-output           Create ready-to-share ZIP bundle (default: True)
  --quiet                Suppress console progress logs
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
