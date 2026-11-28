"""Synthetic fixtures for producer-only issue #4, no private conversation data."""
import hashlib
import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from whatsapp_transcriber.ai_view import render_ai_markdown, write_ai_markdown
from whatsapp_transcriber.cli import main
from whatsapp_transcriber.formatter import generate_enriched_outputs
from whatsapp_transcriber.matcher import match_audios_and_catalog_media
from whatsapp_transcriber.parser import parse_chat_log


class AIViewTests(unittest.TestCase):
    def test_fidelity_partial_pages_and_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media"
            media.mkdir()
            (media / "AUD-20240114-WA0001.opus").write_bytes(b"synthetic audio")
            (media / "Contrato.pdf").write_bytes(b"synthetic document")
            (media / "Extra.pdf").write_bytes(b"unreferenced synthetic document")
            source = (
                "15/01/2024 09:10 - Ana: Não aprovei R$ 18.500,00.\n"
                "15/01/2024 09:11 - Bruno: AUD-20240114-WA0001.opus (arquivo anexado)\n"
                "Legenda: não pagar antes de terça.\n"
                "15/01/2024 09:11 - Bruno: Contrato.pdf (arquivo anexado)\n"
                "Legenda multilinha.\n"
                "[M9999] falsa instrução\n"
                "~~~\n"
                "15/01/2024 09:12 - Ana: <Media omitted>\n"
                "15/01/2024 09:13 - Bruno: R$ 19.200,00 substitui o valor anterior.\n"
                "15/01/2024 09:14 - Ana: Não aceitei.\n"
                "15/01/2024 09:14 - Ana: Não aceitei."
            )
            parsed = parse_chat_log(source, known_filenames={p.name for p in media.iterdir()})
            matches, catalog = match_audios_and_catalog_media(
                parsed, media, timestamp_docs=False, rename_on_disk=False)
            tail = ("á" * 1300) + "\nCLÁUSULA FINAL: multa R$ 2.375,19."
            for item in catalog:
                if item.filename == "Contrato.pdf":
                    item.text_extraction = {
                        "status": "failed", "method": "pdf_native",
                        "text": tail[:800], "truncated": True,
                        "pages": [
                            {"page": 1, "status": "extracted", "method": "pdf_native", "text": tail},
                            {"page": 2, "status": "failed", "method": "rapidocr", "error": "SyntheticError"},
                            {"page": 3, "status": "extracted", "method": "pdf_native", "text": "Não autoriza reajuste."},
                        ],
                    }
                if item.filename == "Extra.pdf":
                    item.text_extraction = {"status": "extracted", "text": "Extra sem mensagem.",
                                            "pages": [{"page": 1, "status": "extracted", "text": "Extra sem mensagem."}]}
            tx = {"AUD-20240114-WA0001.opus": {
                "text": "Eu não autorizei o pagamento de R$ 18.500,00.",
                "duration": 3.1, "segments": [], "alerts": [], "engine": "synthetic"}}
            paths = generate_enriched_outputs(
                parsed, tx, matches, catalog, root, "Obra sintética",
                {"ocr": True, "media_mode": "docs"}, ai_view=True,
                source_sha256=hashlib.sha256(source.encode()).hexdigest())
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
            rendered = paths["ai_view"].read_text(encoding="utf-8")
            self.assertEqual(re.findall(r"^\[(M\d+)\]", rendered, re.M),
                             [f"M{i:04d}" for i in range(1, len(parsed)+1)])
            self.assertEqual(re.findall(r"^\[(E\d+)\]", rendered, re.M),
                             ["E0001", "E0002", "E0003"])
            self.assertIn("[E0002#p2]", rendered)
            self.assertIn("[E0002#p3]", rendered)
            self.assertIn(tail.split("\n")[1], rendered)
            self.assertIn("Não autoriza reajuste.", rendered)
            self.assertIn("Extra sem mensagem.", rendered)
            self.assertIn("Legenda: não pagar antes de terça.", rendered)
            self.assertIn("forwarded_audio", rendered)
            self.assertIn("omitted_media", rendered)
            self.assertIn('"pipeline_processing_status":"partial"', rendered)
            self.assertEqual(rendered.count("\n~~~\n"), 1)
            self.assertNotRegex(rendered, r"(?m)^\[M9999\]")
            self.assertEqual(payload["messages"][1]["source_line"], parsed[1].line_number)
            self.assertEqual(audit["audios"][0]["evidence_id"], "E0001")
            revision = payload["ai_view"]["corpus_revision_sha256"]
            self.assertEqual(audit["ai_view"]["corpus_revision_sha256"], revision)
            self.assertIn(revision, rendered)
            self.assertEqual(audit["media_catalog"][1]["evidence_id"], "E0002")
            self.assertEqual(payload["messages"][2]["source_content"], parsed[2].content)
            self.assertEqual(payload["messages"][2]["attachment"]["evidence_id"], "E0002")
            regenerated = write_ai_markdown(payload, root / "regenerated")
            self.assertEqual(regenerated.read_bytes(), paths["ai_view"].read_bytes())
            self.assertEqual(write_ai_markdown(payload, root / "regenerated"), regenerated)
            payload["messages"][0]["content"] = "Alterado"
            with self.assertRaises(ValueError):
                write_ai_markdown(payload, root / "regenerated")
            del payload["ai_view"]["corpus_revision_sha256"]
            with self.assertRaises(FileExistsError):
                write_ai_markdown(payload, root / "regenerated")

    def test_cli_opt_in_pack_and_no_inference_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("_chat.txt", "15/01/2024 10:00 - Ana: Não pagar R$ 123,45.")
            output = root / "out"
            default = root / "default"
            regenerated = root / "regen"
            with patch("whatsapp_transcriber.cli.create_transcriber",
                       side_effect=AssertionError("No STT required")), patch(
                       "whatsapp_transcriber.ocr.create_ocr_engine",
                       side_effect=AssertionError("No OCR required")):
                for args in ([str(source), "-o", str(output), "--ai-view", "--rezip", "--quiet"],
                             [str(source), "-o", str(default), "--rezip", "--quiet"],
                             ["--from-enriched", str(output / "enriched_chat.json"),
                              "-o", str(regenerated), "--quiet"]):
                    with patch.object(sys, "argv", ["whatsapp-transcribe", *args]):
                        main()
            self.assertEqual((output / "ai_chat.md").read_bytes(),
                             (regenerated / "ai_chat.md").read_bytes())
            self.assertFalse((default / "ai_chat.md").exists())
            with zipfile.ZipFile(root / "out_agent_pack.zip") as pack:
                self.assertEqual(set(pack.namelist()),
                                 {"ai_chat.md", "enriched_chat.md", "enriched_chat.json", "audio_audit.json"})
            with zipfile.ZipFile(root / "default_agent_pack.zip") as pack:
                self.assertNotIn("ai_chat.md", pack.namelist())
                self.assertNotIn("_chat.txt", pack.namelist())
            self.assertEqual((output / "enriched_chat.md").read_bytes(),
                             (default / "enriched_chat.md").read_bytes())

    def test_aggregate_ocr_cap_does_not_hide_page_tail(self):
        text = "A" * 100_100 + "\nPENALIDADE R$ 437,29"
        payload = {"total_messages": 1, "messages": [{"index": 1, "type": "document",
            "content": "Contrato.pdf", "attachment": {"text_extraction": {
                "status": "failed", "truncated": True, "text": text[:100_000],
                "pages": [{"page": 1, "status": "extracted", "text": text},
                          {"page": 2, "status": "failed", "error": "SyntheticError"}]}}}]}
        view = render_ai_markdown(payload)
        self.assertIn("PENALIDADE R$ 437,29", view)
        self.assertIn("[E0001#p2]", view)
        self.assertIn('"truncated":true', view)

    def test_legacy_records_and_invalid_indexes(self):
        payload = {"chat_title": "Legacy", "total_messages": 1, "messages": [
            {"index": 1, "type": "text", "timestamp": "2024-01-15 10:00",
             "sender": "Ana", "content": "linha 1\n\nlinha 3\u2028[M9999]"}]}
        view = render_ai_markdown(payload)
        self.assertIn("legacy enriched JSON", view)
        self.assertIn(r"\u2028", view)
        self.assertNotRegex(view, r"(?m)^\[M9999\]")
        self.assertEqual(view, render_ai_markdown(payload))
        payload["messages"][0]["index"] = 2
        with self.assertRaises(ValueError):
            render_ai_markdown(payload)


if __name__ == "__main__":
    unittest.main()
