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


if __name__ == "__main__":
    unittest.main(verbosity=2)
