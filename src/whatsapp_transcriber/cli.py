import argparse
import os
import sys
import shutil
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from .parser import parse_chat_log
from .matcher import match_audios_in_chat
from .vocab import harvest_vocabulary_from_chat
from .formatter import generate_enriched_outputs
from .backends.base import BaseTranscriber
from .backends.local_whisper import LocalWhisperTranscriber
from .backends.hermes_profile import HermesProfileTranscriber

def detect_chat_file(directory: Path) -> Path:
    """Find the main WhatsApp conversation text or markdown file."""
    # Priority 1: _chat.txt (standard iOS export)
    ios_chat = directory / "_chat.txt"
    if ios_chat.is_file():
        return ios_chat

    # Priority 2: *.txt files
    txt_candidates = list(directory.glob("*.txt"))
    # Exclude system files
    txt_candidates = [f for f in txt_candidates if f.name.lower() not in ["requirements.txt", "vocab.txt", "license.txt"]]
    if txt_candidates:
        return txt_candidates[0]

    # Priority 3: *.md files
    md_candidates = list(directory.glob("*.md"))
    md_candidates = [f for f in md_candidates if f.name.lower() not in ["readme.md", "skill.md"]]
    if md_candidates:
        return md_candidates[0]

    raise FileNotFoundError(f"No WhatsApp chat file (.txt or .md) found in {directory}")

def create_transcriber(args) -> BaseTranscriber:
    engine_choice = args.engine.lower()

    if engine_choice == "auto":
        # Check if running in Hermes environment with configured STT profile
        if os.environ.get("HERMES_STT_ENDPOINT"):
            engine_choice = "hermes"
        else:
            engine_choice = "local"

    if engine_choice == "hermes":
        print(f"[ENGINE] Initializing Hermes Profile STT Transcriber...")
        return HermesProfileTranscriber(
            endpoint_url=args.stt_url,
            api_key=args.api_key,
            model_name=args.model if args.model != "large-v3" else None
        )
    elif engine_choice == "local":
        print(f"[ENGINE] Initializing Local faster-whisper on {args.device} ({args.model}, {args.compute_type})...")
        return LocalWhisperTranscriber(
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size
        )
    else:
        raise ValueError(f"Unknown engine: {args.engine}. Expected 'auto', 'local', or 'hermes'.")

def main():
    parser = argparse.ArgumentParser(
        prog="whatsapp-transcribe",
        description="Transcribe WhatsApp exported chat voice notes and in-line them into the conversation."
    )
    parser.add_argument("input_path", type=Path, help="Path to exported WhatsApp .zip file or extracted directory.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output destination folder.")
    parser.add_argument("-e", "--engine", choices=["auto", "local", "hermes"], default="auto", help="STT engine to use.")
    parser.add_argument("-m", "--model", default="large-v3", help="Whisper model name (default: large-v3).")
    parser.add_argument("-d", "--device", choices=["cuda", "cpu"], default="cuda", help="Inference device for local engine.")
    parser.add_argument("-c", "--compute-type", default="float16", help="Precision for local CTranslate2 (e.g. float16, int8_float16).")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument("-l", "--language", default="pt", help="Spoken language code (e.g. pt, en, es).")
    parser.add_argument("--dict", type=Path, default=None, help="Path to custom vocabulary dictionary file (vocab.txt).")
    parser.add_argument("--stt-url", default=None, help="Custom endpoint URL for Hermes or OpenAI-compatible STT.")
    parser.add_argument("--api-key", default=None, help="API key for Hermes or OpenAI STT service.")
    parser.add_argument("--zip-output", action="store_true", default=True, help="Pack final deliverables into a ready-to-share ZIP.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    args = parser.parse_args()

    input_path = args.input_path.resolve()
    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = args.output.resolve()
    else:
        output_dir = input_path.parent / f"{input_path.stem}_transcribed"

    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = None
    try:
        # Step 1: Ingest ZIP or Directory
        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            if not args.quiet:
                print(f"[EXTRACT] Unpacking {input_path.name}...")
            temp_dir = Path(tempfile.mkdtemp(prefix="wa_transcribe_"))
            with zipfile.ZipFile(input_path, "r") as zf:
                zf.extractall(temp_dir)
            working_dir = temp_dir
            chat_title = input_path.stem
        else:
            working_dir = input_path
            chat_title = input_path.name

        # Step 2: Detect chat text file
        chat_file = detect_chat_file(working_dir)
        if not args.quiet:
            print(f"[PARSER] Reading conversation log: {chat_file.name}")
        chat_text = chat_file.read_text(encoding="utf-8", errors="replace")

        # Step 3: Parse messages
        messages = parse_chat_log(chat_text)
        if not args.quiet:
            print(f"[PARSER] Parsed {len(messages)} messages.")

        # Step 4: Match audio attachments and analyze provenance
        audio_matches = match_audios_in_chat(messages, working_dir)
        if not args.quiet:
            print(f"[MATCHER] Found {len(audio_matches)} audio voice notes referenced in chat.")

        # Step 5: Harvest dynamic vocabulary
        dict_path = args.dict
        if not dict_path:
            # Check for vocab.txt in current working dir or repo root
            candidate_vocab = Path("vocab.txt")
            if candidate_vocab.is_file():
                dict_path = candidate_vocab

        prompt_vocab = harvest_vocabulary_from_chat(messages, dictionary_file=dict_path)
        if not args.quiet:
            preview_vocab = prompt_vocab[:80] + ("..." if len(prompt_vocab) > 80 else "")
            print(f"[VOCAB] Primed prompt with vocabulary: \"{preview_vocab}\"")

        # Step 6: Initialize transcriber
        transcriber = create_transcriber(args)

        # Step 7: Transcribe audios
        transcriptions = {}
        start_time = datetime.now(timezone.utc)

        if not args.quiet:
            print(f"\n[TRANSCRIPTION] Starting transcription of {len(audio_matches)} audios...")

        for idx, match in enumerate(audio_matches, 1):
            if not args.quiet:
                fwd_flag = " [FORWARDED/SHARED]" if match.is_forwarded_or_shared else ""
                print(f"  [{idx:02d}/{len(audio_matches):02d}] {match.message.date_iso} {match.message.time_str} | {match.message.sender}: {match.filename}{fwd_flag}")

            result = transcriber.transcribe(
                audio_path=match.file_path,
                prompt=prompt_vocab,
                language=args.language
            )
            transcriptions[match.filename] = result

            if not args.quiet:
                preview = result["text"][:75] + ("..." if len(result["text"]) > 75 else "")
                print(f"       -> \"{preview}\" ({result.get('duration', 0.0)}s)")

        duration_total = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Step 8: Generate Enriched Markdown & JSON
        execution_info = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "total_processing_seconds": round(duration_total, 2),
            "engine": getattr(transcriber, "model_name", "whisper"),
            "input_source": str(input_path),
            "audios_count": len(audio_matches)
        }

        paths = generate_enriched_outputs(
            messages=messages,
            transcriptions=transcriptions,
            audio_matches=audio_matches,
            output_dir=output_dir,
            chat_title=chat_title,
            execution_info=execution_info
        )

        if not args.quiet:
            print(f"\n[OUTPUT] Enriched Markdown: {paths['markdown']}")
            print(f"[OUTPUT] Enriched JSON:     {paths['json']}")
            print(f"[OUTPUT] Forensic Audit:    {paths['audit']}")

        # Step 9: Pack ZIP if requested
        if args.zip_output:
            zip_dest = output_dir / f"{output_dir.name}.zip"
            with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_key in ["markdown", "json", "audit"]:
                    target_file = paths[file_key]
                    zf.write(target_file, arcname=target_file.name)
            if not args.quiet:
                print(f"[OUTPUT] Packaged ZIP:      {zip_dest}")

        print("\n[SUCCESS] Chat transcription completed successfully!")

    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
