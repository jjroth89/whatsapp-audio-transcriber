import argparse
import os
import sys
import shutil
import zipfile
import tempfile
from pathlib import Path, PurePosixPath
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
OCR_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
DOCS_AND_IMAGE_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".ppt", ".pptx", ".txt", ".odt", ".ods", ".rtf",
    ".dwg", ".dxf", ".rvt", ".ifc", ".skp", ".step", ".stp",
    ".vcf", ".zip", ".rar", ".7z", ".tar", ".gz",
    ".jpg", ".jpeg", ".png", ".gif", ".heic", ".bmp", ".tiff",
}


def _choose_chat_name(filenames: list[str]) -> str:
    """Pick a single WhatsApp export deterministically, never an arbitrary .txt attachment."""
    names = sorted(filenames, key=lambda name: (name.casefold(), name))
    preferred = [name for name in names if name.casefold() == "_chat.txt"]
    if not preferred:
        preferred = [name for name in names if name.casefold().startswith(
            ("whatsapp chat with", "conversa do whatsapp com")
        ) and Path(name).suffix.lower() in {".txt", ".md"}]
    if not preferred:
        preferred = [name for name in names if Path(name).suffix.lower() == ".txt"
                     and name.casefold() not in {"requirements.txt", "vocab.txt", "license.txt"}]
    if not preferred:
        preferred = [name for name in names if Path(name).suffix.lower() == ".md"
                     and name.casefold() not in {"readme.md", "skill.md", "agents.md"}]
    if len(preferred) != 1:
        raise ValueError(f"Expected exactly one WhatsApp chat export, found {len(preferred)} candidates")
    return preferred[0]


def detect_chat_file(directory: Path) -> Path:
    names = [p.name for p in directory.iterdir() if p.is_file()]
    return directory / _choose_chat_name(names)


def _safe_basename(archive_name: str) -> str:
    name = PurePosixPath(archive_name.replace("\\", "/")).name
    if name in ("", ".", ".."):
        raise ValueError(f"Unsafe archive member: {archive_name!r}")
    return name


def copy_folder_input(
    source_dir: Path, output_dir: Path, media_dir: Path,
    media_mode: str, transcribe_videos: bool, ocr: bool = False
) -> Tuple[Path, Set[str]]:
    """Copy selected assets to the output; never rename/delete the source files."""
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Output directory must be outside the source directory")
    source_files = sorted((p for p in source_dir.rglob("*") if p.is_file()),
                          key=lambda path: path.as_posix().casefold())
    chat_file = detect_chat_file(source_dir)
    by_name = {}
    for p in source_files:
        if p == chat_file:
            continue
        key = p.name.casefold()
        if key in by_name:
            raise ValueError(f"Duplicate source basename: {p.name}")
        by_name[key] = p
    media_dir.mkdir(parents=True, exist_ok=True)
    for key, p in by_name.items():
        ext = p.suffix.lower()
        needed = (ext in AUDIO_EXTENSIONS or (transcribe_videos and ext in VIDEO_EXTENSIONS)
                  or (ocr and ext in OCR_EXTENSIONS))
        if needed or _should_retain_ext(ext, p.name, media_mode):
            shutil.copy2(p, media_dir / p.name)
    return chat_file, set(by_name)


def create_agent_pack(output_dir: Path, zip_target: Path) -> Path:
    """Allowlist canonical artifacts and retained media; never include the raw chat export."""
    files = []
    for name in ("enriched_chat.md", "enriched_chat.json", "audio_audit.json"):
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required agent artifact not found: {path}")
        files.append(path)
    media = output_dir / "media"
    if media.exists():
        files.extend(p for p in media.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_target, "w") as zf:
        for path in sorted(files, key=lambda item: item.relative_to(output_dir).as_posix()):
            arc = path.relative_to(output_dir).as_posix()
            info = zipfile.ZipInfo(arc, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with path.open("rb") as src, zf.open(info, "w") as dst:
                shutil.copyfileobj(src, dst)
    return zip_target


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
    ocr: bool = False,
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
        members = [(member, _safe_basename(member.filename))
                   for member in zf.infolist() if not member.is_dir()]
        seen = set()
        for _, name in members:
            key = name.casefold()
            if key in seen:
                raise ValueError(f"Duplicate ZIP basename: {name}")
            seen.add(key)
        chat_name = _choose_chat_name([name for _, name in members])
        for member, fname in members:
            ext = Path(fname).suffix.lower()
            if fname == chat_name:
                target_path = dest_dir / fname
                with zf.open(member) as src, target_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                chat_file_found = target_path
                continue

            available_in_source.add(fname.casefold())
            needed_for_tx = (ext in AUDIO_EXTENSIONS or (transcribe_videos and ext in VIDEO_EXTENSIONS)
                             or (ocr and ext in OCR_EXTENSIONS))
            if needed_for_tx or _should_retain_ext(ext, fname, media_mode):
                target_path = media_dir / fname
                with zf.open(member) as src, target_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    if chat_file_found is None:
        raise FileNotFoundError("WhatsApp chat export not found")

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
    media_grp.add_argument("--rezip", action="store_true", help="Bundle enriched outputs and selected media into <output>_agent_pack.zip.")
    media_grp.add_argument("--ocr", action="store_true", help="Extract PDF text and local JPEG/PNG OCR (requires the [ocr] extra).")

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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty to prevent stale pack contents: {output_dir}")
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
            ocr=args.ocr,
        )
    else:
        chat_file, available_in_source = copy_folder_input(
            input_path, output_dir, media_dir, media_mode, args.transcribe_videos,
            ocr=args.ocr,
        )

    if not args.quiet:
        print(f"[PARSER] Reading conversation: {chat_file.name}")
    chat_text = chat_file.read_text(encoding="utf-8", errors="replace")
    messages = parse_chat_log(chat_text, known_filenames=available_in_source)
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
    transcriber = create_transcriber(args) if audio_matches else None

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

    # OCR works on temporary extracted attachments BEFORE retention policy omits originals.
    if args.ocr:
        from .ocr import create_ocr_engine, extract_media_text
        target_items = [
            item for item in catalog_items
            if item.extracted and (Path(item.saved_filename or item.filename).suffix.lower() in OCR_EXTENSIONS)
        ]
        if target_items:
            ocr_engine = create_ocr_engine()
            ocr_cache = {}
            for item in target_items:
                file_path = media_dir / (item.saved_filename or item.filename)
                # One physical file may be referenced by multiple messages.
                key = str(file_path.resolve())
                if key not in ocr_cache:
                    ocr_cache[key] = extract_media_text(file_path, ocr_engine)
                item.text_extraction = ocr_cache[key]

    duration_total = (datetime.now(timezone.utc) - start_time).total_seconds()

    # Apply media retention policy BEFORE generating outputs so manifests & links are 100% accurate
    purged_av = False
    if True:  # Both ZIP and folder sources use the private output/media copy.
        for cat in catalog_items:
            if not cat.extracted:
                continue
            fpath = media_dir / (cat.saved_filename or cat.filename)
            ext = Path(cat.filename).suffix.lower()
            should_keep = _should_retain_ext(ext, cat.filename, media_mode)
            if not should_keep:
                result = transcriptions.get(cat.saved_filename or "") or transcriptions.get(cat.filename)
                was_transcribed = bool(result and str(result.get("text", "")).strip()
                                       and "no_audio_track" not in result.get("alerts", []))
                if fpath.exists():
                    fpath.unlink()
                cat.extracted = False
                cat.local_rel_path = None
                if was_transcribed:
                    cat.retention_status = "text_equivalent_verified"
                    purged_av = True
                elif cat.attachment_type in ("audio", "video") and (cat.filename in transcriptions or (cat.saved_filename or "") in transcriptions):
                    cat.retention_status = "processing_failed"
                else:
                    cat.retention_status = "not_packaged_by_choice"

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
        "engine": getattr(transcriber, "model_name", "not_required"),
        "media_mode": media_mode,
        "transcribe_videos": args.transcribe_videos,
        "ocr": args.ocr,
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
        create_agent_pack(output_dir, zip_target)
        if not args.quiet:
            print(f"[REZIP] Agent Pack created: {zip_target}")

    if not args.quiet:
        print("\n[SUCCESS] Chat transcription and artifact generation completed successfully!")


if __name__ == "__main__":
    main()
