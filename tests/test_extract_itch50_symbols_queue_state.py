#!/usr/bin/env python3
"""Queue-reactive training artifacts for the streaming ITCH extractor."""

from __future__ import annotations

import argparse
import csv
import gzip
import pathlib
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import extract_itch50_symbol as single  # noqa: E402
import extract_itch50_symbols as extractor  # noqa: E402


LOCATE = 17
SYMBOL = "QQQ"


def clock_ns(value: str, fraction_ns: int = 0) -> int:
    return single.parse_clock_ns(value) + fraction_ns


def header(kind: str, event_time_ns: int) -> bytes:
    return (
        kind.encode("ascii")
        + LOCATE.to_bytes(2, "big")
        + (1).to_bytes(2, "big")
        + event_time_ns.to_bytes(6, "big")
    )


def directory(event_time_ns: int) -> bytes:
    message = header("R", event_time_ns) + b"QQQ     " + bytes(20)
    assert len(message) == 39
    return message


def add(reference: int, side: str, shares: int, price: int,
        event_time_ns: int) -> bytes:
    message = (
        header("A", event_time_ns)
        + reference.to_bytes(8, "big")
        + side.encode("ascii")
        + shares.to_bytes(4, "big")
        + b"QQQ     "
        + price.to_bytes(4, "big")
    )
    assert len(message) == 36
    return message


def execute(reference: int, shares: int, match: int,
            event_time_ns: int) -> bytes:
    message = (
        header("E", event_time_ns)
        + reference.to_bytes(8, "big")
        + shares.to_bytes(4, "big")
        + match.to_bytes(8, "big")
    )
    assert len(message) == 31
    return message


def broken_trade(match: int, event_time_ns: int) -> bytes:
    message = header("B", event_time_ns) + match.to_bytes(8, "big")
    assert len(message) == 19
    return message


def cancel(reference: int, shares: int, event_time_ns: int) -> bytes:
    message = (
        header("X", event_time_ns)
        + reference.to_bytes(8, "big")
        + shares.to_bytes(4, "big")
    )
    assert len(message) == 23
    return message


def delete(reference: int, event_time_ns: int) -> bytes:
    message = header("D", event_time_ns) + reference.to_bytes(8, "big")
    assert len(message) == 19
    return message


def write_itch(path: pathlib.Path, messages: list[bytes]) -> None:
    with gzip.open(path, "wb") as output:
        for message in messages:
            output.write(len(message).to_bytes(2, "big"))
            output.write(message)


def read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def extraction_args(source: pathlib.Path, output: pathlib.Path,
                    targets: pathlib.Path | None) -> argparse.Namespace:
    return argparse.Namespace(
        input=str(source),
        input_sha256="",
        symbols=[SYMBOL],
        date="2020-01-30",
        start="09:30:00",
        end="09:30:04",
        snapshot_ms=1000,
        output_root=str(output),
        progress_seconds=0.0,
        target_window_seconds=[],
        state_targets_csv=str(targets) if targets is not None else None,
    )


class QueueReactiveExtractorTest(unittest.TestCase):
    def fixture_messages(self) -> list[bytes]:
        return [
            directory(clock_ns("04:00:00")),
            add(1, "B", 100, 2_200_000, clock_ns("09:29:59")),
            add(2, "S", 100, 2_200_300, clock_ns("09:29:59", 1)),
            # Side-specific one-tick improvements from a three-tick spread.
            add(3, "B", 25, 2_200_100,
                clock_ns("09:30:00", 500_000_000)),
            add(4, "S", 20, 2_200_200, clock_ns("09:30:01")),
            # The first market-buy observation is removed by B everywhere.
            execute(4, 5, 42, clock_ns("09:30:01", 500_000_000)),
            broken_trade(42, clock_ns("09:30:01", 600_000_000)),
            execute(4, 5, 43, clock_ns("09:30:02")),
            cancel(3, 5, clock_ns("09:30:02", 500_000_000)),
            delete(4, clock_ns("09:30:03")),
        ]

    def test_training_artifacts_conserve_events_exposure_and_reversals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "queue_state.itch.gz"
            output = root / "out"
            targets = root / "state_targets.csv"
            targets.write_text(
                "symbol,target_mean_bid_depth,target_mean_ask_depth\n"
                "QQQ,100,100\n",
                encoding="utf-8",
            )
            write_itch(source, self.fixture_messages())

            manifests = extractor.extract_many(
                extraction_args(source, output, targets)
            )
            self.assertEqual(len(manifests), 1)
            manifest = manifests[0]
            self.assertEqual(manifest["aggregation_duration_seconds"], 4.0)
            training = manifest["queue_reactive_training_artifacts"]
            self.assertEqual(training["schema_version"], 2)
            self.assertTrue(training["training_only"])
            self.assertTrue(training["queue_policy_estimation_ready"])
            self.assertFalse(training["event_stream_retained"])
            self.assertEqual(
                training["event_count_conservation"]["by_event_type"],
                {
                    "limit_buy": 1,
                    "limit_sell": 1,
                    "market_buy": 1,
                    "market_sell": 0,
                    "cancel_bid": 1,
                    "cancel_ask": 1,
                },
            )
            self.assertEqual(
                training["event_count_conservation"]["half_hour_total"], 5
            )
            self.assertEqual(training["exposure"]["total_seconds"], 4.0)
            self.assertEqual(
                training["exposure"]["valid_two_sided_seconds"], 4.0
            )
            self.assertTrue(
                training["pre_event_state_definition"][
                    "equal_timestamp_messages_share_one_left_limit_state"
                ]
            )

            symbol_dir = output / "itch_20200130_qqq"
            self.assertEqual(
                read_rows(
                    symbol_dir / "limit_buy_improvement_distribution.txt"
                ),
                [
                    {
                        "improvement_ticks": "1",
                        "improvement_price_units": "100",
                        "count": "1",
                    },
                ],
            )
            self.assertEqual(
                read_rows(
                    symbol_dir / "limit_sell_improvement_distribution.txt"
                ),
                [
                    {
                        "improvement_ticks": "1",
                        "improvement_price_units": "100",
                        "count": "1",
                    },
                ],
            )

            time_rows = read_rows(
                symbol_dir / "intraday_event_counts.csv"
            )
            self.assertEqual(len(time_rows), 13 * 6)
            first_bin = {
                row["event_type"]: int(row["count"])
                for row in time_rows
                if row["half_hour_bin"] == "0"
            }
            self.assertEqual(
                first_bin,
                training["event_count_conservation"]["by_event_type"],
            )
            self.assertTrue(
                all(
                    row["count"] == "0"
                    for row in time_rows
                    if row["half_hour_bin"] != "0"
                )
            )

            state_rows = read_rows(
                symbol_dir / "queue_state_counts.csv"
            )
            market_rows = [
                row for row in state_rows
                if row["event_type"] == "market_buy"
            ]
            self.assertEqual(len(market_rows), 1)
            self.assertEqual(market_rows[0]["count"], "1")
            self.assertEqual(
                market_rows[0]["queue_imbalance_bin"], "buy_high"
            )
            self.assertEqual(market_rows[0]["spread_bin"], "one_tick")

            exposure_rows = read_rows(
                symbol_dir / "queue_state_exposure.csv"
            )
            self.assertAlmostEqual(
                sum(float(row["exposure_seconds"]) for row in exposure_rows),
                4.0,
            )
            observed_exposures = {
                (
                    row["spread_bin"],
                    row["queue_imbalance_bin"],
                    row["bid_depth_ratio_bin"],
                    row["ask_depth_ratio_bin"],
                ): float(row["exposure_seconds"])
                for row in exposure_rows
            }
            self.assertEqual(
                observed_exposures,
                {
                    ("wider", "balanced", "typical", "typical"): 0.5,
                    ("wider", "sell_high", "low", "typical"): 0.5,
                    ("one_tick", "balanced", "low", "low"): 0.5,
                    ("one_tick", "buy_high", "low", "low"): 1.5,
                    ("wider", "sell_very_high", "low", "typical"): 1.0,
                },
            )

            # Both legacy and new market-flow artifacts retain only match 43.
            self.assertEqual(
                read_rows(symbol_dir / "market_buy_quantity_distribution.txt"),
                [{"quantity": "5", "count": "1"}],
            )
            self.assertEqual(
                manifest["data_quality_counts"][
                    "broken_trade_flow_observations_removed"
                ],
                1,
            )

    def test_legacy_invocation_emits_no_unexposed_queue_policy_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "legacy.itch.gz"
            output = root / "out"
            write_itch(source, self.fixture_messages())

            manifests = extractor.extract_many(
                extraction_args(source, output, None)
            )
            self.assertNotIn(
                "queue_reactive_training_artifacts", manifests[0]
            )
            symbol_dir = output / "itch_20200130_qqq"
            for filename in (
                "limit_buy_improvement_distribution.txt",
                "limit_sell_improvement_distribution.txt",
                "intraday_event_counts.csv",
                "queue_state_counts.csv",
                "queue_state_exposure.csv",
            ):
                self.assertFalse((symbol_dir / filename).exists())

    def test_exposure_is_split_exactly_at_half_hour_boundaries(self) -> None:
        start = clock_ns("09:30:00")
        end = clock_ns("10:30:00")
        training = extractor.QueueReactiveTrainingSummary(
            extractor.StateTargets(100.0, 100.0),
            extractor.StateTargetsSource("targets.csv", "0" * 64),
            start,
            end,
        )
        book = single.VisibleBook()
        book.add(1, "B", 100, 2_200_000)
        book.add(2, "S", 100, 2_200_100)

        training.advance_exposure(book, end)
        self.assertEqual(
            training.state_exposure_ns,
            {
                (0, "one_tick", "balanced", "typical", "typical"):
                    1800 * 1_000_000_000,
                (1, "one_tick", "balanced", "typical", "typical"):
                    1800 * 1_000_000_000,
            },
        )

    def test_equal_timestamp_events_use_one_predictable_left_limit_state(self) -> None:
        start = clock_ns("09:30:00")
        end = clock_ns("09:30:10")
        event_time = start + 1_000_000_000
        training = extractor.QueueReactiveTrainingSummary(
            extractor.StateTargets(100.0, 100.0),
            extractor.StateTargetsSource("targets.csv", "0" * 64),
            start,
            end,
        )
        book = single.VisibleBook()
        book.add(1, "B", 100, 2_200_000)
        book.add(2, "S", 100, 2_200_100)

        training.advance_exposure(book, event_time)
        first = training.observe_event("market_buy", event_time, book)

        # A zero-duration mutation at the same exchange timestamp must not
        # become a new point-process covariate state.
        book.reduce(1, 90)
        training.advance_exposure(book, event_time)
        second = training.observe_event("market_sell", event_time, book)
        self.assertEqual(first.state_key[2:], second.state_key[2:])
        self.assertEqual(
            first.state_key[2:],
            ("one_tick", "balanced", "typical", "typical"),
        )

        # At a strictly later timestamp the post-batch state is the new left
        # limit and has accrued positive exposure.
        later = event_time + 1_000_000_000
        training.advance_exposure(book, later)
        third = training.observe_event("market_sell", later, book)
        self.assertNotEqual(first.state_key[2:], third.state_key[2:])

    def test_state_targets_must_cover_every_selected_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            targets = pathlib.Path(temporary) / "targets.csv"
            targets.write_text(
                "symbol,target_mean_bid_depth,target_mean_ask_depth\n"
                "AAPL,100,200\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "lacks selected symbols: QQQ"):
                extractor.load_state_targets(str(targets), ["QQQ"])


if __name__ == "__main__":
    unittest.main()
