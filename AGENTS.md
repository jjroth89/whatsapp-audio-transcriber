# Agentic Integration Guide: Hermes Agent & ChatGPT Desktop

This document outlines how autonomous AI agents and desktop LLM assistants (such as **Hermes Agent** and **ChatGPT Desktop**) should interact with this repository and consume its outputs.

---

## 1. ChatGPT Desktop Workflow

ChatGPT Desktop cannot run local NVIDIA CUDA neural inference directly inside its cloud sandbox. Therefore, the recommended collaboration pattern is **Local Transcribe → Cloud/Desktop Ingest**:

### Step 1: Run Transcription Locally
Run the tool on your machine (using your local GPU or Hermes Agent):

```bash
whatsapp-transcribe /path/to/exported_chat.zip --rezip
```

This creates an output folder containing:
* `enriched_chat.md` (Full conversation with audio transcripts inlined chronologically)
* `enriched_chat.json` (Structured message array)
* `audio_audit.json` (Detailed media and attachment audit)
* `<chat>_agent_pack.zip` (All output deliverables packaged when using `--rezip`)

### Step 2: Upload to ChatGPT Desktop
Attach `enriched_chat.md` (or the packaged `<chat>_agent_pack.zip`) to a ChatGPT Desktop conversation.

### Step 3: Analysis Prompts for ChatGPT Desktop
Use any of the following prompt templates depending on your goal:

#### Prompt A: Forensic & Contractual Timeline Analysis
> *"I have attached the enriched transcript of a WhatsApp conversation (`enriched_chat.md`), where all voice notes have been transcribed in-line with their exact timestamps and provenance tags.
> Please analyze the chronological flow of negotiations:
> 1. Identify all explicit contractual obligations and verbal commitments made by each party.
> 2. Highlight any discrepancies between voice notes and text messages.
> 3. List any forwarded/shared audios (`[FORWARDED/SHARED AUDIO]`) and explain their relevance.
> 4. Cite the date, timestamp, sender, and audio filename for each key finding."*

#### Prompt B: Tax & Financial Evidence Audit
> *"Examine the attached `enriched_chat.md` focusing on financial, tax, and property registration discussions (e.g., CNO, ITBI, Habite-se, RET, pricing, credit agreements).
> Extract all figures, payment terms, interest rates, and loan discussions, noting who proposed them and whether an agreement was finalized."*

---

## 2. Hermes Agent Skill Workflow

If this repository is installed as a skill in Hermes Agent:

### Agent Decision Matrix
* When a user provides a WhatsApp `.zip` export or a folder with `_chat.txt` and `.opus` voice notes:
  1. Invoke the CLI with the desired extraction mode and packaging:
     ```bash
     whatsapp-transcribe <path_to_zip> -o <output_dir> --rezip
     # Or for multimodal vision/doc analysis:
     whatsapp-transcribe <path_to_zip> -o <output_dir> --no-videos --rezip
     ```
  2. If running inside a container with `HERMES_STT_ENDPOINT` set, the tool will automatically use the profile's STT model.
  3. If running on bare-metal with an NVIDIA GPU, it automatically utilizes CUDA FP16.
  4. Once finished, read `enriched_chat.json` for programmatic querying or `enriched_chat.md` for text reasoning.

### Consuming Structured JSON (`enriched_chat.json`)
Hermes agents can parse `enriched_chat.json` directly. Each message contains:
```json
{
  "index": 12,
  "timestamp": "2025-10-14 08:52",
  "sender": "Alice",
  "type": "audio",
  "content": "Hi Bob, I confirmed the revised project schedule...",
  "audio_meta": {
    "filename": "PTT-20251014-WA0001.opus",
    "sha256": "9752b084e88af...",
    "duration_s": 17.68,
    "is_forwarded_or_shared": false,
    "confidence": 1.0
  }
}
```

* Filter messages by `type == "audio"` to inspect all audio transcripts.
* Filter by `type == "forwarded_audio"` to review forwarded evidence.

---

## 3. General AI Agent Capabilities Summary

| Capability | How Handled |
| :--- | :--- |
| **Multi-speaker attribution** | Extracted from chat export metadata; avoids confusing chat title with audio sender. |
| **Forwarded audio integrity** | Flagged when filename date differs from chat message date; avoids timeline distortion. |
| **Confidence scoring** | Average logprob and silence probability recorded per segment in `audio_audit.json`. |
| **Domain vocabulary** | Text messages are scanned dynamically to prevent name hallucinations. |
