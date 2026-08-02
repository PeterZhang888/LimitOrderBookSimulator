#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Focused tests for fixed-cohort local ITCH extraction provenance."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_local_five_day_workflow as five_day  # noqa: E402
import run_local_itch_universe as local  # noqa: E402


class FixedUniverseProvenanceTest(unittest.TestCase):
    def make_manifest(
        self, root: pathlib.Path,
    ) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, dict[str, object]]:
        fixed = root / "fixed.txt"
        fixed.write_text(" aapl \nQQQ\n", encoding="utf-8")
        symbols = root / "selected.txt"
        rendered = "QQQ\nAAPL\n"
        symbols.write_text(rendered, encoding="utf-8")
        manifest = root / "selection.json"
        value = {
            "schema_version": 1,
            "policy": "nasdaq-common-plus-qqq",
            "mode": "fixed_symbols",
            "max_symbols": None,
            "fixed_symbols_input": {
                "path": str(fixed.resolve()),
                "raw_sha256": hashlib.sha256(fixed.read_bytes()).hexdigest(),
                "normalized_count": 2,
            },
            "selected_symbols": {
                "path": str(symbols.resolve()),
                "count": 2,
                "canonical_sha256": hashlib.sha256(
                    rendered.encode("utf-8")
                ).hexdigest(),
                "canonical_encoding": "UTF-8, one symbol per line, LF terminated",
                "canonical_order": "QQQ first, then lexicographic",
                "ordered_symbols": ["QQQ", "AAPL"],
            },
        }
        manifest.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return fixed, symbols, manifest, value

    def test_fixed_selection_record_binds_raw_and_canonical_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixed, symbols, manifest, value = self.make_manifest(root)
            request = local.requested_selection(fixed.resolve(), None)
            record = local.validate_selection_manifest(manifest, symbols, request)

            self.assertEqual(record["mode"], "fixed_symbols")
            self.assertEqual(
                record["fixed_symbols_input"]["raw_sha256"],
                value["fixed_symbols_input"]["raw_sha256"],
            )
            self.assertEqual(
                record["selected_symbols"]["canonical_sha256"],
                value["selected_symbols"]["canonical_sha256"],
            )
            self.assertEqual(
                record["selected_symbols"]["ordered_symbols"], ["QQQ", "AAPL"]
            )
            self.assertEqual(
                record["provenance"]["sha256"], local.sha256_file(manifest)
            )

    def test_tampered_selected_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixed, symbols, manifest, _ = self.make_manifest(root)
            symbols.write_text("QQQ\nMSFT\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from provenance"):
                local.validate_selection_manifest(
                    manifest, symbols, local.requested_selection(fixed.resolve(), None)
                )

    def test_changed_fixed_input_cannot_reuse_existing_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixed, symbols, manifest, _ = self.make_manifest(root)
            fixed.write_text("QQQ\nMSFT\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw_sha256 differs"):
                local.validate_selection_manifest(
                    manifest, symbols, local.requested_selection(fixed.resolve(), None)
                )

    def test_cap_and_fixed_file_are_cli_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            local.build_parser().parse_args([
                "--itch", "x", "--date", "2020-01-30", "--result-root", "r",
                "--max-symbols", "10", "--fixed-symbols", "fixed.txt",
            ])

    def test_batch_marker_binds_extractor_and_state_target_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = pathlib.Path(temporary) / "marker.json"
            marker.write_text(json.dumps({
                "status": "complete",
                "batch": "batch_00000",
                "archive_sha256": "a" * 64,
                "symbols_sha256": "b" * 64,
                "extractor_sha256": "c" * 64,
                "state_targets_sha256": "d" * 64,
            }))
            arguments = {
                "archive_sha256": "a" * 64,
                "symbols_sha256": "b" * 64,
                "batch_name": "batch_00000",
                "extractor_sha256": "c" * 64,
                "state_targets_sha256": "d" * 64,
            }
            self.assertTrue(local.valid_batch_marker(marker, **arguments))
            arguments["state_targets_sha256"] = "e" * 64
            self.assertFalse(local.valid_batch_marker(marker, **arguments))


class FiveDayFixedUniverseForwardingTest(unittest.TestCase):
    def test_same_fixed_file_is_forwarded_to_every_daily_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            downloads = root / "downloads"
            downloads.mkdir()
            (downloads / "train.gz").write_bytes(b"t")
            (downloads / "heldout.gz").write_bytes(b"h")
            binary = root / "simulator"
            binary.write_bytes(b"binary")
            fixed = root / "fixed.txt"
            fixed.write_text("QQQ\nAAPL\n", encoding="utf-8")
            commands: list[list[str]] = []

            with (
                mock.patch.object(
                    five_day, "TRAINING", (("2019-01-30", "train.gz", 1),)
                ),
                mock.patch.object(
                    five_day, "HELDOUT", ("2020-01-30", "heldout.gz", 1)
                ),
                mock.patch.object(
                    five_day, "run", side_effect=lambda command: commands.append(list(command))
                ),
            ):
                self.assertEqual(five_day.main([
                    "--project-dir", str(PROJECT_ROOT),
                    "--download-dir", str(downloads),
                    "--work-root", str(root / "work"),
                    "--binary", str(binary),
                    "--fixed-symbols", str(fixed),
                    "--no-wait",
                ]), 0)

            extraction_commands = commands[:2]
            self.assertEqual(len(extraction_commands), 2)
            for command in extraction_commands:
                index = command.index("--fixed-symbols")
                self.assertEqual(command[index + 1], str(fixed.resolve()))
            self.assertNotIn("--fixed-symbols", commands[-1])

    def test_declaration_change_during_workflow_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            downloads = root / "downloads"
            downloads.mkdir()
            (downloads / "train.gz").write_bytes(b"t")
            (downloads / "heldout.gz").write_bytes(b"h")
            binary = root / "simulator"
            binary.write_bytes(b"binary")
            fixed = root / "fixed.txt"
            fixed.write_text("QQQ\nAAPL\n", encoding="utf-8")

            def mutate_declaration(_command: object) -> None:
                fixed.write_text("QQQ\nMSFT\n", encoding="utf-8")

            with (
                mock.patch.object(
                    five_day, "TRAINING", (("2019-01-30", "train.gz", 1),)
                ),
                mock.patch.object(
                    five_day, "HELDOUT", ("2020-01-30", "heldout.gz", 1)
                ),
                mock.patch.object(five_day, "run", side_effect=mutate_declaration),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "declaration changed during"
                ):
                    five_day.main([
                        "--project-dir", str(PROJECT_ROOT),
                        "--download-dir", str(downloads),
                        "--work-root", str(root / "work"),
                        "--binary", str(binary),
                        "--fixed-symbols", str(fixed),
                        "--no-wait",
                    ])


if __name__ == "__main__":
    unittest.main()
