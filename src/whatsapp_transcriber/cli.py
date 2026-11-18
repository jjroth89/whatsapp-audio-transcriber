import argparse
import os
import sys
import shutil
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Set, Tuple

from . import __version__
from .parser import parse_chat_log, extract_chat_title
from .matcher import match_audios_and_catalog_media
from .vocab import harvest_vocabulary_from_chat
from .formatter import generate_enriched_outputs
from .hardware import detect_hardware
from .video import extract_audio_from_video
from .backends.base import BaseTranscriber
from .backends.local_whisper import LocalWhisperTranscriber
from .backends.hermes_profile import HermesProfileTranscriber

VIDEO_EXTENSIONS = {".mp4", ".3gp", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".opus", ".ogg", ".m4a", ".mp3", ".wav", ".aac", ".flac"}
STICKER_EXTENSIONS = {".webp"}
DOCS_AND_IMAGE_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".ppt", ".pptx", ".txt", ".odt", ".ods", ".rtf",
    ".dwg", ".dxf", ".rvt", ".ifc", ".skp", ".step", ".stp",
    ".vcf", ".zip", ".rar", ".7z", ".tar", ".gz",
    ".jpg", ".jpeg", ".png", ".gif", ".heic", ".bmp", ".tiff",
}


def detect_chat_file(directory: Path) -> Path:
    """Find the main WhatsApp conversation text or markdown file."""
    ios_chat = directory / "_chat.txt"
    if ios_chat.is_file():
        return ios_chat

    txt_candidates = [
        f for f in directory.glob("*.txt")
        if f.name.lower() not in ["requirements.txt", "vocab.txt", "license.txt"]
    ]
    if txt_candidates:
        return txt_candidates[0]

    md_candidates = [
        f for f in directory.glob("*.md")
        if f.name.lower() not in ["readme.md", "skill.md", "agents.md"]
    ]
    if md_candidates:
        return md_candidates[0]

    raise FileNotFoundError(f"No WhatsApp conversation file (.txt or .md) found in {directory}")


def _should_retain_ext(ext: str, fname: str, media_mode: str) -> bool:
    """Return True if the file should remain in final media/ under media_mode."""
    is_sticker = (ext in STICKER_EXTENSIONS) or fname.upper().startswith("STK-")
    if media_mode == "all":
        return True
    if media_mode == "no-videos":
        return ext not in VIDEO_EXTENSIONS
    if media_mode == "docs":
        if is_sticker or ext in VIDEO_EXTENSIONS or ext in AUDIO_EXTENSIONS:
            return False
        return ext in DOCS_AND_IMAGE_EXTENSIONS
    if media_mode == "audio":
        return ext in AUDIO_EXTENSIONS
    if media_mode == "none":
        return False
    return ext in AUDIO_EXTENSIONS


def selective_unpack(
    zip_path: Path,
    dest_dir: Path,
    media_dir: Path,
    media_mode: str = "audio",
    transcribe_videos: bool = False,
) -> Tuple[Path, Set[str]]:
    """
    Extracts chat log, retained media per media_mode, and temporary A/V files
    needed for transcription. Returns (chat_file_path, all_zip_filenames_lower).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    chat_file_found = None
    available_in_source: Set[str] = set()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            fname = Path(member.filename).name
            ext = Path(fname).suffix.lower()

            if ext in [".txt", ".md"] and not chat_file_found:
                if fname.lower().startswith(("_chat", "whatsapp chat with", "conversa do whatsapp com")) or ext == ".txt":
                    target_path = dest_dir / fname
                    with zf.open(member) as src, open(target_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    chat_file_found = target_path
                    continue

            available_in_source.add(fname.lower())

            # Extract if retained by policy OR needed temporarily for transcription
            needed_for_tx = (ext in AUDIO_EXTENSIONS) or (transcribe_videos and ext in VIDEO_EXTENSIONS)
            if needed_for_tx or _should_retain_ext(ext, fname, media_mode):
                target_path = media_dir / fname
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    if not chat_file_found:
        chat_file_found = detect_chat_file(dest_dir)

    return chat_file_found, available_in_source


def create_transcriber(args) -> BaseTranscriber:
    engine_choice = args.engine.lower()

    if engine_choice == "auto":
        engine_choice = "hermes" if os.environ.get("HERMES_STT_ENDPOINT") else "local"

    if engine_choice == "hermes":
        if not args.quiet:
            print("[ENGINE] Initializing Hermes Profile STT Transcriber...")
        return HermesProfileTranscriber(
            endpoint_url=args.stt_url,
            api_key=args.api_key,
            model_name=args.model if args.model != "large-v3" else None,
        )
    elif engine_choice == "local":
        hw_profile = detect_hardware(args.device, args.compute_type)
        if not args.quiet:
            print(f"[HARDWARE] {hw_profile.details}")

        return LocalWhisperTranscriber(
            model_name=args.model,
            device=hw_profile.device,
            compute_type=hw_profile.compute_type,
            batch_size=args.batch_size,
            cache_dir=args.model_dir,
        )
    else:
        raise ValueError(f"Unknown engine: {args.engine}. Expected 'auto', 'local', or 'hermes'.")


def _resolve_media_mode(args) -> str:
    """Resolve unified --media mode with fallback for hidden deprecated flags."""
    if getattr(args, "lean", False):
        return "none"
    if getattr(args, "extract_all", False):
        return "all"
    if getattr(args, "no_videos", False):
        return "no-videos"
    return args.media


def main():
    parser = argparse.ArgumentParser(
        prog="whatsapp-transcribe",
        description=f"WhatsApp Audio Transcriber v{__version__} — Local-first forensic & agentic transcription with timeline embedding.",
    )

    core_grp = parser.add_argument_group("Core")
    core_grp.add_argument("input_path", type=Path, help="Path to exported WhatsApp .zip file or folder.")
    core_grp.add_argument("-o", "--output", type=Path, default=None, help="Output destination folder.")
    core_grp.add_argument("-l", "--language", default="pt", help="Spoken language code (default: pt).")

    media_grp = parser.add_argument_group("Media & Packaging")
    media_grp.add_argument(
        "--media",
        choices=["audio", "all", "no-videos", "docs", "none"],
        default="audio",
        help=(
            "Media retention policy in output media/: "
            "'audio' (voice notes only, default), "
            "'all' (all media & docs), "
            "'no-videos' (all except video files), "
            "'docs' (PDFs, images, CAD & contacts only; purges A/V after transcription), "
            "'none' (lean mode; artifacts only, purges all raw media)."
        ),
    )
    media_grp.add_argument(
        "--transcribe-videos",
        action="store_true",
        help="Extract & transcribe audio tracks from video files (works even when video binaries are not retained).",
    )
    media_grp.add_argument("--rezip", action="store_true", help="Bundle final output directory into <output>_agent_pack.zip.")

    engine_grp = parser.add_argument_group("Engine & Model")
    engine_grp.add_argument("-e", "--engine", choices=["auto", "local", "hermes"], default="auto", help="STT engine (default: auto).")
    engine_grp.add_argument("-m", "--model", default="large-v3", help="Whisper model name (default: large-v3).")
    engine_grp.add_argument("-d", "--device", choices=["auto", "cuda", "cpu"], default="auto", help="Hardware device (default: auto).")
    engine_grp.add_argument("-c", "--compute-type", default="auto", help="Precision (auto, float16, int8_float16, int8, float32).")
    engine_grp.add_argument("--model-dir", type=Path, default=None, help="Custom folder for loading/caching Whisper weights.")
    engine_grp.add_argument("--dict", type=Path, default=None, help="Custom vocabulary file (vocab.txt).")

    adv_grp = parser.add_argument_group("Advanced / Remote STT")
    adv_grp.add_argument("-b", "--batch-size", type=int, default=8, help="Inference batch size (default: 8).")
    adv_grp.add_argument("--stt-url", default=None, help="Custom endpoint URL for Hermes STT profile.")
    adv_grp.add_argument("--api-key", default=None, help="API key for Hermes or OpenAI STT service.")
    adv_grp.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output.")

    # Hidden backward-compatible aliases for v1.1 scripts
    parser.add_argument("--extract-all", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-videos", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lean", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    input_path = args.input_path.resolve()
    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output.resolve() if args.output else input_path.parent / f"{input_path.stem}_transcribed"
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    media_mode = _resolve_media_mode(args)
    is_zip_input = input_path.is_file() and input_path.suffix.lower() == ".zip"

    chat_title = extract_chat_title(input_path.name)
    available_in_source = None

    if is_zip_input:
        if not args.quiet:
            print(f"[EXTRACT] Unpacking {input_path.name} (media policy: {media_mode}, transcribe_videos: {args.transcribe_videos})...")
        chat_file, available_in_source = selective_unpack(
            zip_path=input_path,
            dest_dir=output_dir,
            media_dir=media_dir,
            media_mode=media_mode,
            transcribe_videos=args.transcribe_videos,
        )
    else:
        chat_file = detect_chat_file(input_path)
        media_dir = input_path

    if not args.quiet:
        print(f"[PARSER] Reading conversation: {chat_file.name}")
    chat_text = chat_file.read_text(encoding="utf-8", errors="replace")
    messages = parse_chat_log(chat_text)
    if not args.quiet:
        print(f"[PARSER] Parsed {len(messages)} messages.")

    audio_matches, catalog_items = match_audios_and_catalog_media(
        messages=messages,
        media_dir=media_dir,
        include_videos=args.transcribe_videos,
        available_in_source=available_in_source,
        timestamp_docs=is_zip_input,
        rename_on_disk=is_zip_input,
    )
    if not args.quiet:
        audio_cnt = sum(1 for m in audio_matches if m.media_kind == "audio")
        video_cnt = sum(1 for m in audio_matches if m.media_kind == "video")
        print(f"[MATCHER] Matched {audio_cnt} voice notes and {video_cnt} video(s) to transcribe ({len(catalog_items)} total attachments cataloged).")

    dict_path = args.dict
    if not dict_path:
        candidate_vocab = Path("vocab.txt")
        if candidate_vocab.is_file():
            dict_path = candidate_vocab

    prompt_vocab = harvest_vocabulary_from_chat(messages, dictionary_file=dict_path)
    transcriber = create_transcriber(args)

    transcriptions = {}
    start_time = datetime.now(timezone.utc)

    if not args.quiet:
        print(f"\n[TRANSCRIPTION] Starting transcription of {len(audio_matches)} item(s)...")

    with tempfile.TemporaryDirectory(prefix="wa_video_tx_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for idx, match in enumerate(audio_matches, 1):
            if not args.quiet:
                fwd_flag = " [FORWARDED/SHARED]" if match.is_forwarded_or_shared else ""
                kind_tag = "[VIDEO]" if match.media_kind == "video" else "[AUDIO]"
                print(f"  [{idx:02d}/{len(audio_matches):02d}] {kind_tag} {match.message.date_iso} {match.message.time_str} | {match.message.sender}: {match.filename}{fwd_flag}")

            target_audio_path = match.file_path
            if match.media_kind == "video":
                wav_tmp = tmp_dir / f"video_{idx}.wav"
                extracted_wav = extract_audio_from_video(match.file_path, wav_tmp)
                if extracted_wav is None:
                    result = {
                        "text": "[Video contains no audio track]",
                        "duration": 0.0,
                        "language": args.language,
                        "confidence": 1.0,
                        "alerts": ["no_audio_track"],
                        "segments": [],
                        "engine": "pyav",
                    }
                    transcriptions[match.filename] = result
                    if not args.quiet:
                        print("       -> [Video contains no audio track]")
                    continue
                target_audio_path = extracted_wav

            result = transcriber.transcribe(
                audio_path=target_audio_path,
                prompt=prompt_vocab,
                language=args.language,
            )
            transcriptions[match.filename] = result

            if not args.quiet:
                preview = result["text"][:75] + ("..." if len(result["text"]) > 75 else "")
                print(f"       -> \"{preview}\" ({result.get('duration', 0.0)}s)")

    duration_total = (datetime.now(timezone.utc) - start_time).total_seconds()

    # Apply media retention policy BEFORE generating outputs so manifests & links are 100% accurate
    purged_av = False
    if is_zip_input:
        for cat in catalog_items:
            if not cat.extracted:
                continue
            fpath = media_dir / (cat.saved_filename or cat.filename)
            ext = Path(cat.filename).suffix.lower()
            should_keep = _should_retain_ext(ext, cat.filename, media_mode)
            if not should_keep:
                was_transcribed = (cat.saved_filename in transcriptions) or (cat.filename in transcriptions)
                if fpath.exists():
                    fpath.unlink()
                cat.extracted = False
                cat.local_rel_path = None
                if was_transcribed:
                    cat.retention_status = "transcribed_and_purged"
                    purged_av = True
                else:
                    cat.retention_status = "cataloged_only_not_extracted"

        # Ensure no unreferenced temporary A/V files remain if excluded by media_mode
        if media_dir.exists() and media_mode != "none":
            for leftover in list(media_dir.iterdir()):
                if leftover.is_file() and not _should_retain_ext(leftover.suffix.lower(), leftover.name, media_mode):
                    leftover.unlink()

        if media_mode == "none" and media_dir.exists():
            shutil.rmtree(media_dir, ignore_errors=True)
        elif media_dir.exists() and not any(media_dir.iterdir()):
            media_dir.rmdir()

    execution_info = {
        "version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_processing_seconds": round(duration_total, 2),
        "engine": getattr(transcriber, "model_name", "whisper"),
        "media_mode": media_mode,
        "transcribe_videos": args.transcribe_videos,
        "raw_av_purged_after_transcription": purged_av,
        "audios_count": sum(1 for m in audio_matches if m.media_kind == "audio"),
        "videos_transcribed_count": sum(1 for m in audio_matches if m.media_kind == "video"),
    }

    paths = generate_enriched_outputs(
        messages=messages,
        transcriptions=transcriptions,
        audio_matches=audio_matches,
        catalog_items=catalog_items,
        output_dir=output_dir,
        chat_title=chat_title,
        execution_info=execution_info,
    )

    if not args.quiet:
        print(f"\n[OUTPUT] Enriched Markdown: {paths['markdown']}")
        print(f"[OUTPUT] Enriched JSON:     {paths['json']}")
        print(f"[OUTPUT] Media Audit:       {paths['audit']}")

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


if __name__ == "__main__":
    main()
