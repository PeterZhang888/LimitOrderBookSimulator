#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import generate_source_manifest as manifest  # noqa: E402


class SourceManifestTest(unittest.TestCase):
    def test_excludes_generated_trees_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "src").mkdir()
            (root / "src" / "model.cpp").write_text("model\n", encoding="utf-8")
            (root / "build-local").mkdir()
            (root / "build-local" / "binary").write_text("x", encoding="utf-8")
            (root / "results").mkdir()
            (root / "results" / "run.csv").write_text("x", encoding="utf-8")
            output = root / "SOURCE_MANIFEST.sha256"

            manifest.run(root, output)
            first = output.read_text(encoding="utf-8")
            manifest.run(root, output)

            self.assertEqual(first, output.read_text(encoding="utf-8"))
            self.assertIn("./src/model.cpp", first)
            self.assertNotIn("build-local", first)
            self.assertNotIn("results", first)
            self.assertNotIn("SOURCE_MANIFEST.sha256", first)


if __name__ == "__main__":
    unittest.main()
