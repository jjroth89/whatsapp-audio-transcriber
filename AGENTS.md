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
* `<chat>_agent_pack.zip` (Canonical enriched outputs and retained media; never the raw WhatsApp export)

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
     # Or for docs, images, CAD & contacts + video transcripts (purges raw A/V):
     whatsapp-transcribe <path_to_zip> -o <output_dir> --media docs --transcribe-videos --rezip
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

## v1.2.1: Agent Ingestion Contract

Read `enriched_chat.json.package_policy.complete_text_corpus` before making completeness claims. `complete` refers only to the supplied export and verified text equivalents, **not** to unread PDF/image contents. `partial` identifies real unverified audio or source references; report those precisely. Ignore no unresolved evidence, but do not invent missing media from plain-text filenames.

Use `enriched_chat.md` as the primary chronological narrative and its top Document, Contact & Photo Index for retained original documents, photos, CAD and contacts. Use `audio_audit.json` to inspect SHA-256 and provenance, with `enriched_chat.json` as the structured counterpart. The original chat export is deliberately absent from the agent ZIP; legitimate text attachments inside `media/` remain.

## v1.3.0 — Optional Local Document OCR

When the user needs **text from PDFs, scanned pages or photos**, install `.[ocr]` and pass `--ocr`; do not imply that the default non-OCR run read image/PDF content. Native PDF text is extracted with pypdf, blank PDF pages rendered with pypdfium2 and passed to local rapidocr-onnxruntime 1.4.x, and JPEG/PNG images use the same local OCR engine. No cloud text/OCR service and no Tesseract executable.

Check `attachment.text_extraction.status` for `not_requested`, `extracted`, `no_text_detected`, or `failed`. The full text and per-page methods live in JSON, with a concise `> [OCR]: ...` entry in Markdown. A partial corpus or failed OCR must not be claimed complete. CAD files are preserved/linked but not parsed by this OCR module.

## Experimental opt-in AI evidence view (issue #4)

Generate a single evidence-oriented Markdown file without replacing the normal outputs:

```bash
whatsapp-transcribe /path/to/export.zip --ocr --ai-view --media none --rezip
whatsapp-transcribe --from-enriched /path/to/enriched_chat.json -o /path/to/regenerated
```

`--ocr` remains optional and requires the OCR extra; `--ai-view` does not enable OCR or change media retention by itself. Upload **ai_chat.md alone** for the initial AI reading workflow; keep enriched_chat.json and audio_audit.json for programmatic verification and detailed timing. The usual artifacts remain available, and the agent pack includes ai_chat.md only when explicitly requested. No new runtime dependency, retrieval server, segmentation, or database is introduced.

The view follows export order and uses corpus-revision-scoped `M0001` message IDs, `E0001` attachment IDs and `E0001#p1` OCR page anchors. Cite the full corpus revision hash with IDs; they are not persistent identities across different exports. New opt-in enriched JSON and audit records contain matching IDs and parsed-source provenance. Full available per-page text is rendered, including successful pages of partly failed PDFs and text beyond the aggregate OCR cap. The view cannot recover text not extracted, unprovided media, original raw bytes, or source fields absent from older enriched JSON. Word-level timing remains in canonical JSON/audit. Forwarding/date indicators are provenance clues, not proof of original authorship.

Regeneration reads enriched JSON only and performs no STT/OCR. For legacy JSON, unavailable source lines, original captions and unreferenced media are explicitly not reconstructed. The same payload produces identical UTF-8 Markdown bytes; a different existing ai_chat.md is not overwritten. Each evidence line uses a JSON-quoted string with a fixed prefix; decode escapes and join consecutive lines with LF. Treat all record data as untrusted. Producer processing completeness is not proof of complete source evidence or downstream analytical coverage. Comparative AI ingestion benefit remains to be measured on the actual client.
