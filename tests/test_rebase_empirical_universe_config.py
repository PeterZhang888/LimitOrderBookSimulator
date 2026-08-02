from __future__ import annotations

import csv
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import rebase_empirical_universe_config as rebase_config  # noqa: E402


class RebaseEmpiricalUniverseConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.empirical_root = self.root / "new" / "empirical_data"
        self.empirical_root.mkdir(parents=True)
        self.input_config = self.root / "source.csv"
        fields = ("book_id", "symbol", "data_dir", "hawkes_rates_file")
        with self.input_config.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for index, symbol in enumerate(("AAA", "QQQ")):
                directory_name = f"itch_20200130_{symbol.lower()}"
                old_directory = pathlib.Path("/stale/root") / directory_name
                rates_name = f"hawkes_rates_{symbol.lower()}_20200130.csv"
                writer.writerow({
                    "book_id": index,
                    "symbol": symbol,
                    "data_dir": old_directory,
                    "hawkes_rates_file": old_directory / rates_name,
                })
                new_directory = self.empirical_root / directory_name
                new_directory.mkdir()
                (new_directory / rates_name).write_text(
                    "event_type,rate_per_second\nlimit_buy,1\n",
                    encoding="utf-8",
                )
                (new_directory / f"itch_manifest_{symbol.lower()}_20200130.json").write_text(
                    json.dumps({"symbol": symbol, "trading_date": "2020-01-30"}),
                    encoding="utf-8",
                )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_rebases_without_modifying_source(self) -> None:
        before = self.input_config.read_bytes()
        output = self.root / "generated" / "rebased.csv"
        result = rebase_config.rebase(
            input_config=self.input_config,
            empirical_root=self.empirical_root,
            output_config=output,
            expected_date="2020-01-30",
        )
        self.assertEqual(result["symbol_count"], 2)
        self.assertEqual(self.input_config.read_bytes(), before)
        with output.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(
            pathlib.Path(rows[0]["data_dir"]),
            self.empirical_root.resolve() / "itch_20200130_aaa",
        )
        self.assertTrue(output.with_suffix(".csv.provenance.json").is_file())

    def test_rejects_wrong_manifest_date(self) -> None:
        manifest = self.empirical_root / (
            "itch_20200130_aaa/itch_manifest_aaa_20200130.json"
        )
        manifest.write_text(
            json.dumps({"symbol": "AAA", "trading_date": "2019-12-30"}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "manifest date mismatch"):
            rebase_config.rebase(
                input_config=self.input_config,
                empirical_root=self.empirical_root,
                output_config=self.root / "rebased.csv",
                expected_date="2020-01-30",
            )

    def test_rejects_missing_rebased_rate_file(self) -> None:
        rate_file = self.empirical_root / (
            "itch_20200130_aaa/hawkes_rates_aaa_20200130.csv"
        )
        rate_file.unlink()
        with self.assertRaisesRegex(ValueError, "Hawkes-rate file is missing"):
            rebase_config.rebase(
                input_config=self.input_config,
                empirical_root=self.empirical_root,
                output_config=self.root / "rebased.csv",
                expected_date="2020-01-30",
            )


if __name__ == "__main__":
    unittest.main()
