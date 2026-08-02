from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import prepare_heldout_target_config as target_config  # noqa: E402


class HeldoutTargetConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.target_root = self.root / "cluster_targets"
        self.target_root.mkdir()
        self.source = self.root / "source.csv"
        fields = ("book_id", "symbol", "data_dir")
        with self.source.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for index, symbol in enumerate(("AAA", "QQQ")):
                name = f"itch_20200130_{symbol.lower()}"
                writer.writerow({
                    "book_id": index,
                    "symbol": symbol,
                    "data_dir": f"/private/tmp/old/{name}",
                })
                directory = self.target_root / name
                directory.mkdir()
                (directory / f"market_targets_{symbol.lower()}_20200130.csv").write_text(
                    "name,target,scale\nmean_spread_ticks,1,1\n",
                    encoding="utf-8",
                )
                (directory / f"itch_manifest_{symbol.lower()}_20200130.json").write_text(
                    json.dumps({"symbol": symbol, "trading_date": "2020-01-30"}),
                    encoding="utf-8",
                )
        self.provenance = self.root / "pooling_provenance.json"
        self.write_provenance()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def source_hash(self) -> str:
        return hashlib.sha256(self.source.read_bytes()).hexdigest()

    def write_provenance(self) -> None:
        self.provenance.write_text(json.dumps({
            "heldout_date": "2020-01-30",
            "common_symbol_count": 2,
            "heldout": {
                "heldout_role": "opening_state_and_validation_targets_only",
                "background_inputs_inherited_from_pooled": True,
                "source_config": str(self.source),
                "source_config_sha256": self.source_hash(),
                "target_root": str(self.target_root),
            },
        }), encoding="utf-8")

    def test_rebases_stale_paths_without_modifying_source(self) -> None:
        before = self.source.read_bytes()
        output = self.root / "result" / "targets.csv"
        result = target_config.prepare(
            provenance_path=self.provenance,
            output_path=output,
            expected_date="2020-01-30",
        )
        self.assertEqual(result["symbol_count"], 2)
        self.assertEqual(self.source.read_bytes(), before)
        with output.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(
            pathlib.Path(rows[0]["target_data_dir"]).resolve(),
            (self.target_root / "itch_20200130_aaa").resolve(),
        )
        self.assertTrue(output.with_suffix(".csv.provenance.json").is_file())

    def test_rejects_source_config_hash_mismatch(self) -> None:
        self.source.write_text(self.source.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            target_config.prepare(
                provenance_path=self.provenance,
                output_path=self.root / "targets.csv",
                expected_date="2020-01-30",
            )

    def test_rejects_wrong_manifest_date(self) -> None:
        manifest = self.target_root / (
            "itch_20200130_aaa/itch_manifest_aaa_20200130.json"
        )
        manifest.write_text(
            json.dumps({"symbol": "AAA", "trading_date": "2019-12-30"}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "manifest date mismatch"):
            target_config.prepare(
                provenance_path=self.provenance,
                output_path=self.root / "targets.csv",
                expected_date="2020-01-30",
            )


if __name__ == "__main__":
    unittest.main()
