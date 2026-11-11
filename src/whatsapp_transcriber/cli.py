import argparse
import os
import sys
import shutil
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Set

from . import __version__
from .parser import parse_chat_log, extract_chat_title, classify_attachment
from .matcher import match_audios_and_catalog_media
from .vocab import harvest_vocabulary_from_chat
from .formatter import generate_enriched_outputs
from .hardware import detect_hardware, resolve_model_cache_dir
from .backends.base import BaseTranscriber
from .backends.local_whisper import LocalWhisperTranscriber
from .backends.hermes_profile import HermesProfileTranscriber

VIDEO_EXTENSIONS = {".mp4", ".3gp", ".mov", ".avi", ".mkv"}
AUDIO_EXTENSIONS = {".opus", ".ogg", ".m4a", ".mp3", ".wav", ".aac", ".flac"}

def detect_chat_file(directory: Path) -> Path:
    """Find the main WhatsApp conversation text or markdown file."""
    # Priority 1: _chat.txt (iOS)
    ios_chat = directory / "_chat.txt"
    if ios_chat.is_file():
        return ios_chat

    # Priority 2: *.txt files
    txt_candidates = list(directory.glob("*.txt"))
    txt_candidates = [f for f in txt_candidates if f.name.lower() not in ["requirements.txt", "vocab.txt", "license.txt"]]
    if txt_candidates:
        return txt_candidates[0]

    # Priority 3: *.md files
    md_candidates = list(directory.glob("*.md"))
    md_candidates = [f for f in md_candidates if f.name.lower() not in ["readme.md", "skill.md", "agents.md"]]
    if md_candidates:
        return md_candidates[0]

    raise FileNotFoundError(f"No WhatsApp conversation file (.txt or .md) found in {directory}")

def selective_unpack(
    zip_path: Path,
    dest_dir: Path,
    media_dir: Path,
    mode: str = "standard"  # "standard", "all", "no-videos"
) -> Path:
    """
    Selectively extracts files from a WhatsApp export zip based on mode.
    - 'standard': Extracts only chat log + audio files.
    - 'all': Extracts everything.
    - 'no-videos': Extracts everything except video files.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    chat_file_found = None

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            fname = Path(member.filename).name
            ext = Path(fname).suffix.lower()

            # Chat log detection
            if ext in [".txt", ".md"] and not chat_file_found:
                if fname.lower().startswith(("_chat", "whatsapp chat with", "conversa do whatsapp com")) or ext == ".txt":
                    target_path = dest_dir / fname
                    with zf.open(member) as src, open(target_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    chat_file_found = target_path
                    continue

            # Filtering media
            should_extract = False
            if mode == "all":
                should_extract = True
            elif mode == "no-videos":
                should_extract = (ext not in VIDEO_EXTENSIONS)
            else:
                # Standard (audio-only)
                should_extract = (ext in AUDIO_EXTENSIONS)

            if should_extract:
                target_path = media_dir / fname
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    if not chat_file_found:
        chat_file_found = detect_chat_file(dest_dir)

    return chat_file_found

def create_transcriber(args) -> BaseTranscriber:
    engine_choice = args.engine.lower()

    if engine_choice == "auto":
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
        hw_profile = detect_hardware(args.device, args.compute_type)
        if not args.quiet:
            print(f"[HARDWARE] {hw_profile.details}")

        return LocalWhisperTranscriber(
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
            cache_dir=args.model_dir
        )
    else:
        raise ValueError(f"Unknown engine: {args.engine}. Expected 'auto', 'local', or 'hermes'.")

def main():
    parser = argparse.ArgumentParser(
        prog="whatsapp-transcribe",
        description=f"WhatsApp Audio Transcriber v{__version__} - Forensic and agentic audio transcription with timeline embedding."
    )
    parser.add_argument("input_path", type=Path, help="Path to exported WhatsApp .zip file or folder.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output destination folder.")
    parser.add_argument("-e", "--engine", choices=["auto", "local", "hermes"], default="auto", help="STT engine (default: auto).")
    parser.add_argument("-m", "--model", default="large-v3", help="Whisper model name (default: large-v3).")
    parser.add_argument("-d", "--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device (auto detects CUDA GPU or falls back to CPU).")
    parser.add_argument("-c", "--compute-type", default="auto", help="Precision (auto, float16, int8_float16, int8, float32).")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument("-l", "--language", default="pt", help="Language code (e.g. pt, en, es).")
    parser.add_argument("--model-dir", type=Path, default=None, help="Custom folder for loading/caching Whisper model weights.")
    parser.add_argument("--dict", type=Path, default=None, help="Custom dictionary file for specialized terms (vocab.txt).")
    parser.add_argument("--stt-url", default=None, help="Custom endpoint URL for Hermes STT profile.")
    parser.add_argument("--api-key", default=None, help="API key for Hermes or OpenAI STT service.")

    # Extraction & Media Retention modes
    parser.add_argument("--extract-all", action="store_true", help="Unpack and preserve all media (photos, videos, documents, audio).")
    parser.add_argument("--no-videos", action="store_true", help="Unpack all media and documents except large video files.")
    parser.add_argument("--lean", action="store_true", help="Purge raw audio files after transcription, leaving only the generated artifacts.")

    # Packaging
    parser.add_argument("--rezip", action="store_true", default=False, help="Rezip the final output folder into a compact agent bundle.")
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
    media_dir = output_dir / "media"

    # Determine extraction mode
    if args.extract_all:
        extract_mode = "all"
    elif args.no_videos:
        extract_mode = "no-videos"
    else:
        extract_mode = "standard"

    temp_extract_dir = None
    try:
        chat_title = extract_chat_title(input_path.name)

        # Step 1: Ingest ZIP or Directory
        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            if not args.quiet:
                print(f"[EXTRACT] Unpacking {input_path.name} (mode: {extract_mode})...")
            chat_file = selective_unpack(
                zip_path=input_path,
                dest_dir=output_dir,
                media_dir=media_dir,
                mode=extract_mode
            )
        else:
            chat_file = detect_chat_file(input_path)
            # If input is a directory, use its media directly
            media_dir = input_path

        # Step 2: Parse messages
        if not args.quiet:
            print(f"[PARSER] Reading conversation: {chat_file.name}")
        chat_text = chat_file.read_text(encoding="utf-8", errors="replace")
        messages = parse_chat_log(chat_text)
        if not args.quiet:
            print(f"[PARSER] Parsed {len(messages)} messages.")

        # Step 3: Match audios and catalog all attachments
        audio_matches, catalog_items = match_audios_and_catalog_media(messages, media_dir)
        if not args.quiet:
            print(f"[MATCHER] Found {len(audio_matches)} voice notes to transcribe ({len(catalog_items)} total attachments cataloged).")

        # Step 4: Harvest dynamic vocabulary
        dict_path = args.dict
        if not dict_path:
            candidate_vocab = Path("vocab.txt")
            if candidate_vocab.is_file():
                dict_path = candidate_vocab

        prompt_vocab = harvest_vocabulary_from_chat(messages, dictionary_file=dict_path)

        # Step 5: Initialize transcriber
        transcriber = create_transcriber(args)

        # Step 6: Transcribe audios
        transcriptions = {}
        start_time = datetime.now(timezone.utc)

        if not args.quiet:
            print(f"\n[TRANSCRIPTION] Starting transcription of {len(audio_matches)} voice notes...")

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

        # Step 7: Generate Enriched Markdown & JSON
        execution_info = {
            "version": __version__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "total_processing_seconds": round(duration_total, 2),
            "engine": getattr(transcriber, "model_name", "whisper"),
            "extract_mode": extract_mode,
            "lean": args.lean,
            "audios_count": len(audio_matches)
        }

        paths = generate_enriched_outputs(
            messages=messages,
            transcriptions=transcriptions,
            audio_matches=audio_matches,
            catalog_items=catalog_items,
            output_dir=output_dir,
            chat_title=chat_title,
            execution_info=execution_info
        )

        # Step 8: Handle Lean Mode (Purge extracted audios if requested)
        if args.lean:
            if not args.quiet:
                print("[LEAN] Purging raw audio files to retain only final artifacts...")
            if media_dir.exists():
                shutil.rmtree(media_dir, ignore_errors=True)

        if not args.quiet:
            print(f"\n[OUTPUT] Enriched Markdown: {paths['markdown']}")
            print(f"[OUTPUT] Enriched JSON:     {paths['json']}")
            print(f"[OUTPUT] Media Audit:       {paths['audit']}")

        # Step 9: Rezip output if requested
        if args.rezip:
            zip_target = output_dir.parent / f"{output_dir.name}_agent_pack.zip"
            if not args.quiet:
                print(f"[REZIP] Packaging final deliverables into {zip_target.name}...")
            with zipfile.ZipFile(zip_target, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(output_dir):
                    for f in files:
                        fpath = Path(root) / f
                        arcname = fpath.relative_to(output_dir)
                        zf.write(fpath, arcname=str(arcname))
            if not args.quiet:
                print(f"[REZIP] Agent Pack created: {zip_target}")

        if not args.quiet:
            print("\n[SUCCESS] Chat transcription and artifact generation completed successfully!")

    finally:
        pass

if __name__ == "__main__":
    main()
