"""End-to-end CLI tests for `python -m linda_probe`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestLindaProbeCLI(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.docs_dir = self.tmp_dir / "docs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        (self.docs_dir / "invoice_2026.txt").write_text(
            "Refund policy: return within 14 days.\nInvoice total: 100 USD.\n",
            encoding="utf-8",
        )
        (self.docs_dir / "notes_alpha.md").write_text(
            "# Architecture Notes\nLocal-first search with SQLite FTS5 and FAISS.\n",
            encoding="utf-8",
        )
        # Exact duplicate pair (>= 1024 bytes)
        payload = ("Linda Probe duplicate test payload.\n" * 40).encode("utf-8")
        (self.docs_dir / "dup_a.txt").write_bytes(payload)
        (self.docs_dir / "dup_b.txt").write_bytes(payload)

        self.write_dir = self.tmp_dir / "state"
        self.write_dir.mkdir(parents=True, exist_ok=True)
        self.cfg_path = self.tmp_dir / "localdocs_config.json"

        self.env = os.environ.copy()
        self.env["LOCALDOCS_WRITE_DIR"] = str(self.write_dir)
        self.env["LOCALDOCS_CONFIG"] = str(self.cfg_path)
        self.env["LOCALDOCS_OCR_WHEN"] = "never"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "linda_probe", *args],
            cwd=str(REPO_ROOT),
            env=self.env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )

    def test_version(self) -> None:
        r = self.run_cli("--version")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("linda-probe 2.0.0", r.stdout)

    def test_find_json_and_formats(self) -> None:
        r = self.run_cli("find", "invoice", "--root", str(self.docs_dir), "--format", "json")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        hits = data.get("results", data) if isinstance(data, dict) else data
        self.assertTrue(any("invoice_2026.txt" in str(h) for h in hits), data)

        for fmt in ("table", "csv", "md"):
            rf = self.run_cli("find", "notes", "--root", str(self.docs_dir), "--format", fmt)
            self.assertEqual(rf.returncode, 0, rf.stderr)
            self.assertIn("notes_alpha.md", rf.stdout)

    def test_index_list_and_search(self) -> None:
        ri = self.run_cli("index", str(self.docs_dir), "--format", "json")
        self.assertEqual(ri.returncode, 0, ri.stderr)

        rl = self.run_cli("list", "--format", "json")
        self.assertEqual(rl.returncode, 0, rl.stderr)
        self.assertIn("docs", rl.stdout.lower())

        rs = self.run_cli("search", str(self.docs_dir), "refund", "--format", "json")
        self.assertEqual(rs.returncode, 0, rs.stderr)
        self.assertIn("invoice_2026", rs.stdout)

    def test_dupes(self) -> None:
        r = self.run_cli("dupes", str(self.docs_dir), "--min-size", "100", "--format", "json")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        groups = data.get("groups", [])
        self.assertEqual(len(groups), 1, data)

    def test_settings_roundtrip(self) -> None:
        r1 = self.run_cli("settings", "show", "--format", "json")
        self.assertEqual(r1.returncode, 0, r1.stderr)
        cfg = json.loads(r1.stdout)
        self.assertIn("fuzzy_names", cfg)

        r2 = self.run_cli("settings", "set", "fuzzy_names", "false", "--format", "json")
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertEqual(json.loads(r2.stdout)["fuzzy_names"], False)

        r3 = self.run_cli("settings", "get", "fuzzy_names")
        self.assertEqual(r3.returncode, 0, r3.stderr)
        self.assertEqual(r3.stdout.strip().lower(), "false")

        r4 = self.run_cli("settings", "reset")
        self.assertEqual(r4.returncode, 0, r4.stderr)

    def test_models_and_doctor(self) -> None:
        rm = self.run_cli("models", "--format", "json")
        self.assertEqual(rm.returncode, 0, rm.stderr)
        models = json.loads(rm.stdout)
        self.assertTrue(any(m.get("id") == "linda-ocr-hf" for m in models))

        rd = self.run_cli("doctor", "--json")
        self.assertIn(rd.returncode, (0, 1))
        checks = json.loads(rd.stdout)
        self.assertIsInstance(checks, list)
        self.assertGreater(len(checks), 0)

    def test_missing_folder_returns_nonzero(self) -> None:
        r = self.run_cli("index", str(self.tmp_dir / "nonexistent_dir"))
        self.assertNotEqual(r.returncode, 0)

    def test_docx_and_zip_index_search(self) -> None:
        import zipfile

        docx_path = self.docs_dir / "contract_sample.docx"
        with zipfile.ZipFile(docx_path, "w") as zf:
            zf.writestr(
                "[Content_Types].xml",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                "</Types>",
            )
            zf.writestr(
                "word/document.xml",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body><w:p><w:r><w:t>UniqueDocxToken7741 confidentiality clause.</w:t></w:r></w:p></w:body>"
                "</w:document>",
            )

        zip_path = self.docs_dir / "archive_bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("nested/secret_note.txt", "UniqueZipToken8852 quarterly budget report.")

        ri = self.run_cli("index", str(self.docs_dir), "--format", "json")
        self.assertEqual(ri.returncode, 0, ri.stderr)

        r_docx = self.run_cli("search", str(self.docs_dir), "UniqueDocxToken7741", "--format", "json")
        self.assertEqual(r_docx.returncode, 0, r_docx.stderr)
        self.assertIn("contract_sample.docx", r_docx.stdout)

        r_zip = self.run_cli("search", str(self.docs_dir), "UniqueZipToken8852", "--format", "json")
        self.assertEqual(r_zip.returncode, 0, r_zip.stderr)
        self.assertIn("archive_bundle.zip", r_zip.stdout)

    def test_ocr_and_image_index_search(self) -> None:
        import shutil
        from PIL import Image, ImageDraw, ImageFont

        has_tesseract = bool(
            shutil.which("tesseract")
            or Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").is_file()
        )
        if not has_tesseract:
            try:
                import rapidocr_onnxruntime  # noqa: F401
            except ImportError:
                self.skipTest("Neither tesseract nor rapidocr_onnxruntime is installed")

        img_path = self.docs_dir / "invoice_scan.png"
        img = Image.new("RGB", (900, 220), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 44)
        except Exception:
            font = ImageFont.load_default()
        draw.text((30, 50), "INVOICE 9931 WARSAW", fill="black", font=font)
        draw.text((30, 120), "TOTAL AMOUNT 4200 PLN", fill="black", font=font)
        img.save(img_path)

        ro = self.run_cli("ocr", str(img_path), "--lang", "eng", "--format", "json")
        self.assertEqual(ro.returncode, 0, ro.stderr)
        self.assertIn("9931", ro.stdout)

        ri = self.run_cli("index", str(self.docs_dir), "--ocr", "--format", "json")
        self.assertEqual(ri.returncode, 0, ri.stderr)

        rs = self.run_cli("search", str(self.docs_dir), "WARSAW", "--format", "json")
        self.assertEqual(rs.returncode, 0, rs.stderr)
        self.assertIn("invoice_scan.png", rs.stdout, f"index={ri.stdout}, search={rs.stdout}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
