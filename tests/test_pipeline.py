import unittest
import tempfile
import json
import sys
from pathlib import Path

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from whatsapp_transcriber.parser import parse_chat_log, ChatMessage
from whatsapp_transcriber.matcher import match_audios_in_chat
from whatsapp_transcriber.vocab import harvest_vocabulary_from_chat
from whatsapp_transcriber.formatter import generate_enriched_outputs

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
            matches = match_audios_in_chat(messages, media_dir)

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
            self.assertEqual(len(json_data), 6)
            audio_msg = json_data[2]
            self.assertEqual(audio_msg["type"], "audio")
            self.assertEqual(audio_msg["audio_meta"]["filename"], "PTT-20240115-WA0001.opus")

if __name__ == "__main__":
    unittest.main()
