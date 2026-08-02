#!/usr/bin/env python3
"""Small deterministic integration test for the streaming ITCH extractor."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pathlib
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import extract_itch50_symbol as extractor  # noqa: E402
import extract_itch50_symbols as multi_extractor  # noqa: E402


LOCATE = 17


def clock_ns(value: str, fraction_ns: int = 0) -> int:
    return extractor.parse_clock_ns(value) + fraction_ns


def header(kind: str, event_time_ns: int) -> bytes:
    return (
        kind.encode("ascii")
        + LOCATE.to_bytes(2, "big")
        + (1).to_bytes(2, "big")
        + event_time_ns.to_bytes(6, "big")
    )


def stock_directory(event_time_ns: int) -> bytes:
    message = header("R", event_time_ns) + b"QQQ     " + bytes(20)
    assert len(message) == 39
    return message


def add(reference: int, side: str, shares: int, price: int, event_time_ns: int) -> bytes:
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


def execute(reference: int, shares: int, event_time_ns: int, match_number: int = 99) -> bytes:
    message = (
        header("E", event_time_ns)
        + reference.to_bytes(8, "big")
        + shares.to_bytes(4, "big")
        + match_number.to_bytes(8, "big")
    )
    assert len(message) == 31
    return message


def broken_trade(match_number: int, event_time_ns: int) -> bytes:
    message = header("B", event_time_ns) + match_number.to_bytes(8, "big")
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


def replace(old_reference: int,
            new_reference: int,
            shares: int,
            price: int,
            event_time_ns: int) -> bytes:
    message = (
        header("U", event_time_ns)
        + old_reference.to_bytes(8, "big")
        + new_reference.to_bytes(8, "big")
        + shares.to_bytes(4, "big")
        + price.to_bytes(4, "big")
    )
    assert len(message) == 35
    return message


def write_binary_file(path: pathlib.Path, messages: list[bytes]) -> None:
    with gzip.open(path, "wb") as output:
        for message in messages:
            output.write(len(message).to_bytes(2, "big"))
            output.write(message)


def rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


class ItchExtractorTest(unittest.TestCase):
    def test_target_scales_use_delete_block_jackknife_for_full_sessions(self) -> None:
        summary = extractor.FixedClockSummary()
        for block in range(8):
            for offset in range(300):
                mid = 10_000.0 + block * 10.0 + (offset % 2)
                summary.observations.append((
                    1.0 + 0.1 * block,
                    100.0 + 1_000.0 * block,
                    200.0 + 800.0 * block,
                    mid,
                ))
        scales, method, blocks = summary.target_scales()
        self.assertEqual(method, "delete_block_jackknife_300_observations_with_floors")
        self.assertEqual(blocks, 8)
        self.assertGreater(scales["mean_bid_depth"], 100.0)
        self.assertGreaterEqual(scales["mean_spread_ticks"], 0.25)
        self.assertGreaterEqual(scales["return_variance"], 1.0e-12)
        self.assertEqual(scales["two_sided_sample_fraction"], 0.005)

    def test_window_targets_use_a_matching_contiguous_prefix(self) -> None:
        summary = extractor.FixedClockSummary()
        for index in range(12):
            summary.observations.append((
                2.0 + 0.1 * index,
                100.0 + index,
                200.0 + 2 * index,
                10_000.0 + index,
            ))
        values, scales, method, blocks = summary.window_values_and_scales(5)
        self.assertAlmostEqual(values["mean_spread_ticks"], 2.2)
        self.assertEqual(values["two_sided_sample_fraction"], 1.0)
        self.assertEqual(method, "provisional_10pct_with_metric_floors_short_window")
        self.assertEqual(blocks, 1)
        self.assertGreaterEqual(scales["mean_spread_ticks"], 0.25)
        with self.assertRaisesRegex(ValueError, "fewer fixed-clock"):
            summary.window_values_and_scales(13)

    def test_one_sided_clock_observations_are_retained_as_a_target(self) -> None:
        summary = extractor.FixedClockSummary()
        valid = (1.0, 100.0, 120.0, 10_000.0)
        summary.clock_observations.extend([valid, None, valid, valid])
        summary.observations.extend([valid, valid, valid])
        summary.snapshots = 3
        summary.invalid_snapshots = 1
        values, scales, _, _ = summary.window_values_and_scales(4)
        self.assertEqual(values["two_sided_sample_fraction"], 0.75)
        self.assertGreater(scales["two_sided_sample_fraction"], 0.0)

    def test_multi_symbol_extractor_routes_books_in_one_pass(self) -> None:
        def local_header(kind: str, locate: int, event_time_ns: int) -> bytes:
            return (
                kind.encode("ascii")
                + locate.to_bytes(2, "big")
                + (1).to_bytes(2, "big")
                + event_time_ns.to_bytes(6, "big")
            )

        def directory(symbol: str, locate: int) -> bytes:
            message = (
                local_header("R", locate, clock_ns("04:00:00"))
                + symbol.ljust(8).encode("ascii")
                + bytes(20)
            )
            self.assertEqual(len(message), 39)
            return message

        def local_add(symbol: str, locate: int, reference: int,
                      side: str, shares: int, price: int,
                      event_time_ns: int) -> bytes:
            message = (
                local_header("A", locate, event_time_ns)
                + reference.to_bytes(8, "big")
                + side.encode("ascii")
                + shares.to_bytes(4, "big")
                + symbol.ljust(8).encode("ascii")
                + price.to_bytes(4, "big")
            )
            self.assertEqual(len(message), 36)
            return message

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "two_books.itch.gz"
            output = root / "data"
            messages = [
                directory("QQQ", 17),
                directory("AAPL", 18),
                local_add("QQQ", 17, 1, "B", 100, 2_200_000,
                          clock_ns("09:29:59")),
                local_add("QQQ", 17, 2, "S", 100, 2_200_200,
                          clock_ns("09:29:59", 1)),
                local_add("AAPL", 18, 1, "B", 200, 3_200_000,
                          clock_ns("09:29:59", 2)),
                local_add("AAPL", 18, 2, "S", 300, 3_200_200,
                          clock_ns("09:29:59", 3)),
                local_add("QQQ", 17, 3, "B", 20, 2_200_100,
                          clock_ns("09:30:00", 100)),
                local_add("AAPL", 18, 3, "S", 25, 3_200_300,
                          clock_ns("09:30:00", 200)),
            ]
            write_binary_file(source, messages)
            manifests = multi_extractor.extract_many(argparse.Namespace(
                input=str(source),
                input_sha256="",
                symbols=["QQQ", "AAPL"],
                date="2020-01-30",
                start="09:30:00",
                end="09:30:03",
                snapshot_ms=1000,
                output_root=str(output),
                progress_seconds=0.0,
                target_window_seconds=[2],
            ))

            self.assertEqual([item["symbol"] for item in manifests], ["QQQ", "AAPL"])
            self.assertEqual([item["stock_locate"] for item in manifests], [17, 18])
            self.assertEqual([item["valid_snapshots"] for item in manifests], [3, 3])
            self.assertEqual(
                manifests[0]["placement_counts"],
                {
                    "improvement_eligible_limit_orders": 1,
                    "inside_spread_limit_orders": 1,
                },
            )
            self.assertEqual(
                rows(output / "itch_20200130_basket" / "opening_bbo_20200130.csv"),
                [
                    {
                        "symbol": "QQQ", "clock": "09:30:00",
                        "best_bid_ticks": "2200000", "best_ask_ticks": "2200200",
                        "best_bid_depth": "100", "best_ask_depth": "100",
                        "mid_price_ticks": "2200100.0",
                    },
                    {
                        "symbol": "AAPL", "clock": "09:30:00",
                        "best_bid_ticks": "3200000", "best_ask_ticks": "3200200",
                        "best_bid_depth": "200", "best_ask_depth": "300",
                        "mid_price_ticks": "3200100.0",
                    },
                ],
            )
            self.assertTrue(
                (output / "itch_20200130_qqq"
                 / "market_targets_qqq_20200130_window_2s.csv").is_file()
            )

    def test_multi_symbol_skip_invalid_openings_writes_a_successful_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "one_sided.itch.gz"
            output = root / "data"
            messages = [
                stock_directory(clock_ns("04:00:00")),
                add(1, "B", 100, 2_200_000, clock_ns("09:29:59")),
            ]
            write_binary_file(source, messages)

            manifests = multi_extractor.extract_many(argparse.Namespace(
                input=str(source),
                input_sha256="",
                symbols=["QQQ"],
                date="2020-01-30",
                start="09:30:00",
                end="09:30:03",
                snapshot_ms=1000,
                output_root=str(output),
                progress_seconds=0.0,
                skip_invalid_openings=True,
            ))

            self.assertEqual(manifests, [])
            self.assertEqual(
                rows(output / "itch_20200130_basket" / "opening_bbo_20200130.csv"),
                [],
            )
            self.assertEqual(
                rows(output / "itch_20200130_exclusions.csv"),
                [{"symbol": "QQQ", "reason": "not_two_sided_at_09:30:00"}],
            )

    def test_reconstructs_orders_and_writes_weighted_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "tiny.itch.gz"
            output = root / "out"
            messages = [
                stock_directory(clock_ns("04:00:00")),
                add(1, "B", 100, 2_200_000, clock_ns("09:29:59")),
                add(2, "S", 200, 2_200_200, clock_ns("09:29:59", 1)),
                add(3, "B", 50, 2_199_900, clock_ns("09:30:00", 500_000_000)),
                execute(2, 5, clock_ns("09:30:00", 750_000_000), match_number=100),
                broken_trade(100, clock_ns("09:30:00", 800_000_000)),
                execute(2, 40, clock_ns("09:30:01")),
                cancel(3, 10, clock_ns("09:30:01", 500_000_000)),
                replace(2, 4, 100, 2_200_300, clock_ns("09:30:02")),
                delete(3, clock_ns("09:30:02", 500_000_000)),
            ]
            write_binary_file(source, messages)

            manifest = extractor.extract(argparse.Namespace(
                input=str(source),
                output_dir=str(output),
                symbol="QQQ",
                date="2020-01-30",
                start="09:30:00",
                end="09:30:03",
                snapshot_ms=1000,
                progress_seconds=0.0,
            ))

            self.assertEqual(manifest["stock_locate"], LOCATE)
            self.assertEqual(manifest["total_messages"], len(messages))
            self.assertEqual(manifest["valid_snapshots"], 3)
            self.assertEqual(manifest["invalid_snapshots"], 0)
            self.assertEqual(
                manifest["data_quality_counts"]["broken_trade_flow_observations_removed"],
                1,
            )
            self.assertEqual(
                manifest["distribution_observation_counts"],
                {
                    "limit_buy": 1,
                    "limit_sell": 1,
                    "market_buy": 1,
                    "market_sell": 0,
                    "cancel_bid": 2,
                    "cancel_ask": 1,
                },
            )
            self.assertEqual(
                manifest["placement_counts"],
                {"improvement_eligible_limit_orders": 1},
            )

            self.assertEqual(
                rows(output / "market_buy_quantity_distribution.txt"),
                [{"quantity": "40", "count": "1"}],
            )
            self.assertEqual(
                rows(output / "cancel_bid_quantity_distribution.txt"),
                [{"quantity": "10", "count": "1"}, {"quantity": "40", "count": "1"}],
            )
            self.assertEqual(
                rows(output / "cancel_bid_distance_distribution.txt"),
                [{"distance_ticks": "1", "count": "2"}],
            )
            self.assertEqual(
                rows(output / "cancel_ask_distance_distribution.txt"),
                [{"distance_ticks": "0", "count": "1"}],
            )

            manifest_path = output / "itch_manifest_qqq_20200130.json"
            with manifest_path.open() as source_file:
                written = json.load(source_file)
            self.assertEqual(written["symbol"], "QQQ")
            self.assertEqual(written["snapshot_interval_ms"], 1000)


if __name__ == "__main__":
    unittest.main()
