import unittest
import tempfile
import json
import sys
import zipfile
from unittest.mock import patch
from pathlib import Path

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from whatsapp_transcriber.parser import parse_chat_log, ChatMessage, resolve_attachment
from whatsapp_transcriber.matcher import match_audios_in_chat, match_audios_and_catalog_media
from whatsapp_transcriber.vocab import harvest_vocabulary_from_chat
from whatsapp_transcriber.formatter import generate_enriched_outputs
from whatsapp_transcriber.vcf import parse_vcf_file
from whatsapp_transcriber.video import extract_audio_from_video
from whatsapp_transcriber.cli import create_agent_pack, copy_folder_input, selective_unpack, main

class TestWhatsAppAudioTranscriber(unittest.TestCase):

    def setUp(self):
        self.sample_android_chat = """15/01/2024 12:50 - Messages and calls are end-to-end encrypted.
15/01/2024 12:52 - Alice: Good afternoon, Bob! How are you?
15/01/2024 12:57 - Alice: PTT-20240115-WA0001.opus (file attached)
15/01/2024 14:55 - Bob: This proposal looks solid to me.
Let's finalize with the NDA and MOU clauses included.
15/10/2025 14:30 - Bob: AUD-20240312-WA0002.opus (file attached)
15/10/2025 14:31 - Bob: <Forwarded message>
PTT-20230510-WA0003.opus (file attached)
"""

        self.sample_ios_chat = """[15/01/2024, 12:52:10] Alice: Good afternoon Bob!
[15/01/2024, 12:57:15] Alice: <attached: PTT-20240115-WA0001.opus>
[15/01/2024, 14:55:00] Bob: Confirmed!
"""

    def test_android_parser(self):
        messages = parse_chat_log(self.sample_android_chat)
        self.assertEqual(len(messages), 6)

        # Check system message
        self.assertTrue(messages[0].is_system)

        # Check native audio message
        msg_audio = messages[2]
        self.assertEqual(msg_audio.sender, "Alice")
        self.assertEqual(msg_audio.attachment_filename, "PTT-20240115-WA0001.opus")
        self.assertFalse(msg_audio.is_forwarded)

        # Check forwarded audio message
        msg_fwd = messages[5]
        self.assertEqual(msg_fwd.attachment_filename, "PTT-20230510-WA0003.opus")
        self.assertTrue(msg_fwd.is_forwarded)

    def test_ios_parser(self):
        messages = parse_chat_log(self.sample_ios_chat)
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1].attachment_filename, "PTT-20240115-WA0001.opus")

    def test_matcher_provenance(self):
        messages = parse_chat_log(self.sample_android_chat)
        with tempfile.TemporaryDirectory() as tmpdir:
            media_dir = Path(tmpdir)
            # Create mock audio files
            f1 = media_dir / "PTT-20240115-WA0001.opus"
            f2 = media_dir / "AUD-20240312-WA0002.opus"
            f3 = media_dir / "PTT-20230510-WA0003.opus"
            f1.write_bytes(b"dummy audio content 1")
            f2.write_bytes(b"dummy audio content 2")
            f3.write_bytes(b"dummy audio content 3")

            matches = match_audios_in_chat(messages, media_dir)
            self.assertEqual(len(matches), 3)

            # Native audio
            m1 = matches[0]
            self.assertEqual(m1.filename, "PTT-20240115-WA0001.opus")
            self.assertFalse(m1.is_forwarded_or_shared)
            self.assertEqual(m1.origin_date, "2024-01-15")

            # Shared AUD file (different date: 2024-03-12 vs chat date 2025-10-15)
            m2 = matches[1]
            self.assertEqual(m2.filename, "AUD-20240312-WA0002.opus")
            self.assertTrue(m2.is_forwarded_or_shared)
            self.assertEqual(m2.origin_date, "2024-03-12")

            # Forwarded PTT file with older date
            m3 = matches[2]
            self.assertEqual(m3.filename, "PTT-20230510-WA0003.opus")
            self.assertTrue(m3.is_forwarded_or_shared)
            self.assertEqual(m3.origin_date, "2023-05-10")

    def test_vocabulary_harvester(self):
        messages = parse_chat_log(self.sample_android_chat)
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tf:
            tf.write("Acquisition, Compliance, Audit\n")
            dict_path = Path(tf.name)

        try:
            vocab = harvest_vocabulary_from_chat(messages, dictionary_file=dict_path)
            self.assertIn("Acquisition", vocab)
            self.assertIn("Compliance", vocab)
            self.assertIn("NDA", vocab)
            self.assertIn("MOU", vocab)
            self.assertIn("Alice", vocab)
        finally:
            dict_path.unlink(missing_ok=True)

    def test_formatter_outputs(self):
        messages = parse_chat_log(self.sample_android_chat)
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            media_dir = work_dir / "media"
            out_dir = work_dir / "output"
            media_dir.mkdir()

            f1 = media_dir / "PTT-20240115-WA0001.opus"
            f1.write_bytes(b"dummy")
            matches, catalog = match_audios_and_catalog_media(messages, media_dir, rename_on_disk=False)

            transcriptions = {
                "PTT-20240115-WA0001.opus": {
                    "text": "Hello Bob, proposal accepted as discussed.",
                    "duration": 24.5,
                    "language": "en",
                    "confidence": 1.0,
                    "segments": [],
                    "alerts": [],
                    "engine": "mock-whisper"
                }
            }

            paths = generate_enriched_outputs(
                messages=messages,
                transcriptions=transcriptions,
                audio_matches=matches,
                catalog_items=catalog,
                output_dir=out_dir,
                chat_title="Test Chat"
            )

            # Verify files exist
            self.assertTrue(paths["markdown"].is_file())
            self.assertTrue(paths["json"].is_file())
            self.assertTrue(paths["audit"].is_file())

            # Verify Markdown contains inlined audio
            md_text = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("[AUDIO NOTE | Duration: 24.5s | File: [`PTT-20240115-WA0001.opus`](media/PTT-20240115-WA0001.opus)]", md_text)
            self.assertIn("Hello Bob, proposal accepted as discussed.", md_text)

            # Verify JSON contains structured message objects
            json_data = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(len(json_data["messages"]), 6)
            audio_msg = json_data["messages"][2]
            self.assertEqual(audio_msg["type"], "audio")
            self.assertEqual(audio_msg["audio_meta"]["filename"], "PTT-20240115-WA0001.opus")

    def test_spaced_filenames_cad_vcf_omitted_edited(self):
        complex_chat = """15/01/2024 09:10 - Alice: Projeto Estrutural (Rev 02).dwg (file attached)
15/01/2024 09:12 - Alice: Orçamento Final - Aprovado.pdf (file attached)
15/01/2024 09:15 - Bob: João Silva.vcf (file attached)
15/01/2024 09:20 - Bob: <Media omitted>
15/01/2024 09:25 - Alice: Updated scope confirmed. <This message was edited>
"""
        messages = parse_chat_log(complex_chat)
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[0].attachment_filename, "Projeto Estrutural (Rev 02).dwg")
        self.assertEqual(messages[0].attachment_type, "cad")
        self.assertEqual(messages[1].attachment_filename, "Orçamento Final - Aprovado.pdf")
        self.assertEqual(messages[1].attachment_type, "document")
        self.assertEqual(messages[2].attachment_filename, "João Silva.vcf")
        self.assertEqual(messages[2].attachment_type, "contact")
        self.assertTrue(messages[3].is_omitted_media)
        self.assertTrue(messages[4].is_edited)

    def test_vcf_parser(self):
        vcard = """BEGIN:VCARD
VERSION:2.1
FN;CHARSET=UTF-8;ENCODING=QUOTED-PRINTABLE:Jo=C3=A3o Silva
TEL;CELL:+55 41 99999-0000
ORG:Acme Engineering
EMAIL:joao@example.com
END:VCARD
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "contact.vcf"
            p.write_text(vcard, encoding="utf-8")
            parsed = parse_vcf_file(p)
            self.assertEqual(parsed["name"], "João Silva")
            self.assertEqual(parsed["phones"], ["+55 41 99999-0000"])
            self.assertEqual(parsed["organization"], "Acme Engineering")
            self.assertEqual(parsed["emails"], ["joao@example.com"])

    def test_timestamp_renaming_and_purged_retention_manifest(self):
        chat = """15/01/2024 10:05 - Alice: Planta Baixa.dwg (file attached)
15/01/2024 10:06 - Alice: PTT-20240115-WA0001.opus (file attached)
"""
        messages = parse_chat_log(chat)
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            media_dir = work_dir / "media"
            out_dir = work_dir / "out"
            media_dir.mkdir()

            dwg = media_dir / "Planta Baixa.dwg"
            dwg.write_bytes(b"dwg-bytes")
            opus = media_dir / "PTT-20240115-WA0001.opus"
            opus.write_bytes(b"opus-bytes")

            matches, catalog = match_audios_and_catalog_media(messages, media_dir, rename_on_disk=True)

            # Non-timestamped CAD file should now be timestamped on disk
            expected_dwg = media_dir / "2024-01-15_1005_Planta Baixa.dwg"
            self.assertTrue(expected_dwg.is_file())
            self.assertEqual(catalog[0].saved_filename, "2024-01-15_1005_Planta Baixa.dwg")
            self.assertEqual(catalog[0].local_rel_path, "media/2024-01-15_1005_Planta Baixa.dwg")

            # Simulate post-transcription purge of audio per docs/none policy
            opus.unlink()
            catalog[1].extracted = False
            catalog[1].local_rel_path = None
            catalog[1].retention_status = "text_equivalent_verified"

            transcriptions = {
                "PTT-20240115-WA0001.opus": {
                    "text": "Confirmed.",
                    "duration": 2.0,
                    "language": "en",
                    "confidence": 1.0,
                    "segments": [],
                    "alerts": [],
                    "engine": "mock-whisper",
                }
            }
            paths = generate_enriched_outputs(
                messages=messages,
                transcriptions=transcriptions,
                audio_matches=matches,
                catalog_items=catalog,
                output_dir=out_dir,
                chat_title="Retention Test",
                execution_info={"media_mode": "docs", "transcribe_videos": False, "raw_av_purged_after_transcription": True},
            )

            md_text = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("media/2024-01-15_1005_Planta%20Baixa.dwg", md_text)
            self.assertIn("text equivalent verified; original not included in pack", md_text)
            self.assertNotIn("(media/PTT-20240115-WA0001.opus)", md_text)

            json_data = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(json_data["package_policy"]["schema_version"], 2)
            self.assertEqual(json_data["package_policy"]["complete_text_corpus"]["status"], "complete")
            self.assertEqual(json_data["package_policy"]["evidence_categories"]["text_equivalent_verified"], 1)
            self.assertEqual(json_data["package_policy"]["evidence_categories"]["retained_original"], 1)


    def test_plain_filename_mentions_require_evidence(self):
        chat = """15/01/2024 10:00 - Alice: Send contract.docx tomorrow.
15/01/2024 10:01 - Bob: The extension is .pdf.
15/01/2024 10:02 - Charlie: Proposal.pdf (file attached)
15/01/2024 10:03 - Alice: Notes.txt
"""
        messages = parse_chat_log(chat)
        self.assertEqual([m.attachment_filename for m in messages], [
            None, None, "Proposal.pdf", None
        ])
        self.assertEqual(messages[2].attachment_evidence, "explicit_marker")
        with tempfile.TemporaryDirectory() as tmp:
            matches, catalog = match_audios_and_catalog_media(messages, Path(tmp))
            self.assertFalse(matches)
            self.assertEqual(len(catalog), 1)
            self.assertEqual(catalog[0].retention_status, "source_reference_unavailable")

    def test_exact_source_filename_matching_and_multiline_marker(self):
        chat = """15/01/2024 10:00 - Alice: Bob, please inspect plan v2.pdf later.
15/01/2024 10:01 - Bob: See attached note:
Orçamento Final.pdf (arquivo anexado)
"""
        messages = parse_chat_log(chat, known_filenames={"plan v2.pdf"})
        self.assertEqual(messages[0].attachment_filename, "plan v2.pdf")
        self.assertEqual(messages[0].attachment_evidence, "source_filename_match")
        self.assertEqual(messages[1].attachment_filename, "Orçamento Final.pdf")
        self.assertEqual(messages[1].attachment_evidence, "explicit_marker")

    def test_source_duplicate_zip_basename_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work / "synthetic.zip"
            with zipfile.ZipFile(source, "w") as zf:
                zf.writestr("_chat.txt", "15/01/2024 10:00 - Alice: Hello")
                zf.writestr("first/Proposal.pdf", b"pdf-a")
                zf.writestr("second/proposal.pdf", b"pdf-b")
            with self.assertRaisesRegex(ValueError, "Duplicate ZIP basename"):
                selective_unpack(source, work / "out", work / "out/media")

    def test_agent_pack_allows_media_txt_but_not_raw_chat(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            out = work / "out"
            (out / "media").mkdir(parents=True)
            for name in ("enriched_chat.md", "enriched_chat.json", "audio_audit.json"):
                (out / name).write_text("synthetic", encoding="utf-8")
            (out / "_chat.txt").write_text("raw private chat placeholder", encoding="utf-8")
            (out / "media/Notes.txt").write_text("synthetic notes", encoding="utf-8")
            archive = create_agent_pack(out, work / "pack.zip")
            with zipfile.ZipFile(archive) as zf:
                self.assertEqual(zf.namelist(), [
                    "audio_audit.json", "enriched_chat.json", "enriched_chat.md", "media/Notes.txt"
                ])
                self.assertEqual(zf.read("media/Notes.txt"), b"synthetic notes")
                self.assertNotIn("_chat.txt", zf.namelist())

    def test_folder_source_is_unchanged_and_media_copy_is_packaged(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work / "input"
            out = work / "output"
            source.mkdir()
            out.mkdir()
            (source / "_chat.txt").write_text(
                "15/01/2024 10:00 - Alice: Proposal.pdf (file attached)", encoding="utf-8"
            )
            (source / "Proposal.pdf").write_bytes(b"synthetic-pdf")
            chat, known = copy_folder_input(source, out, out / "media", "docs", False)
            self.assertEqual(chat, source / "_chat.txt")
            self.assertEqual(known, {"proposal.pdf"})
            self.assertEqual((out / "media/Proposal.pdf").read_bytes(), b"synthetic-pdf")
            self.assertEqual((source / "Proposal.pdf").read_bytes(), b"synthetic-pdf")
            with self.assertRaisesRegex(ValueError, "outside the source"):
                copy_folder_input(source, source / "output", source / "output/media", "all", False)

    def test_complete_corpus_is_truthful_and_document_index_links_resolve(self):
        chat = """15/01/2024 09:10 - Alice: Plan (Rev 2).dwg (file attached)
15/01/2024 09:15 - Bob: Proposal.pdf (file attached)
15/01/2024 09:20 - Charlie: PTT-20240115-WA0001.opus (file attached)
"""
        messages = parse_chat_log(chat)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            media = out / "media"
            media.mkdir()
            (media / "Proposal.pdf").write_bytes(b"synthetic")
            (media / "Plan (Rev 2).dwg").write_bytes(b"synthetic")
            _, catalog = match_audios_and_catalog_media(messages, media, rename_on_disk=False)
            paths = generate_enriched_outputs(
                messages, {}, [], catalog, out, chat_title="Synthetic"
            )
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["package_policy"]["complete_text_corpus"]["status"], "partial")
            self.assertEqual(payload["package_policy"]["complete_text_corpus"]["audio_references_total"], 1)
            md = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("Document, Contact & Photo Index", md)
            self.assertIn("media/Plan%20%28Rev%202%29.dwg", md)
            self.assertIn("media/Proposal.pdf", md)

    def test_text_only_zip_cli_does_not_initialize_stt_and_excludes_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work / "synthetic.zip"
            with zipfile.ZipFile(source, "w") as zf:
                zf.writestr("_chat.txt", "15/01/2024 10:00 - Alice: Hello Bob")
                zf.writestr("Notes.txt", b"synthetic notes")
            with patch.object(sys, "argv", ["whatsapp-transcribe", str(source),
                                           "--media", "docs", "--rezip", "--quiet"]):
                with patch("whatsapp_transcriber.cli.create_transcriber",
                           side_effect=AssertionError("STT must not start")):
                    main()
            output = work / "synthetic_transcribed"
            archive = work / "synthetic_transcribed_agent_pack.zip"
            with zipfile.ZipFile(archive) as zf:
                names = zf.namelist()
                self.assertIn("enriched_chat.md", names)
                self.assertIn("media/Notes.txt", names)
                self.assertNotIn("_chat.txt", names)
            self.assertTrue((output / "_chat.txt").exists())

    def test_nonempty_output_dir_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work / "_chat.txt"
            source.write_text("15/01/2024 10:00 - Alice: Hi")
            out = work / "output"
            out.mkdir()
            (out / "stale.json").write_text("{}")
            with patch.object(sys, "argv", ["whatsapp-transcribe", str(work),
                                           "--output", str(out), "--quiet"]):
                with self.assertRaises(FileExistsError):
                    main()

    def test_video_audio_extraction(self):
        import av
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mp4_path = tmp / "sample.mp4"
            wav_path = tmp / "out.wav"

            # Generate a tiny silent/sine mp4 with an AAC audio stream using PyAV
            with av.open(str(mp4_path), mode="w") as container:
                astream = container.add_stream("aac", rate=44100)
                astream.layout = "mono"
                samples = np.zeros((1, 1024), dtype=np.float32)
                for _ in range(5):
                    frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
                    frame.sample_rate = 44100
                    for pkt in astream.encode(frame):
                        container.mux(pkt)
                for pkt in astream.encode(None):
                    container.mux(pkt)

            result = extract_audio_from_video(mp4_path, wav_path)
            self.assertIsNotNone(result)
            self.assertTrue(wav_path.is_file())
            self.assertGreater(wav_path.stat().st_size, 44)

if __name__ == "__main__":
    unittest.main()
