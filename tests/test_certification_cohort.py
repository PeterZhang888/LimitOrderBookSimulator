#!/usr/bin/env python3
"""Regression tests for the immutable 1,480-symbol identity contract."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "certification_cohort_under_test",
    ROOT / "scripts" / "certification_cohort.py",
)
assert SPEC is not None and SPEC.loader is not None
COHORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COHORT
SPEC.loader.exec_module(COHORT)


class CertificationCohortTest(unittest.TestCase):
    def setUp(self) -> None:
        self.symbols = COHORT.load_required_symbols(ROOT)

    def test_bundled_cohort_has_exact_identity_and_origin(self) -> None:
        cohort_path, manifest_path = COHORT.project_paths(ROOT)
        self.assertEqual(len(self.symbols), 1480)
        self.assertEqual(self.symbols[0], "QQQ")
        self.assertEqual(
            self.symbols[1:],
            tuple(sorted(symbol for symbol in self.symbols if symbol != "QQQ")),
        )
        self.assertEqual(
            hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
            "2f57f37762772d9523fb9916fe2376a9578e337d20971fe39aa44d578f5691d3",
        )
        self.assertTrue(cohort_path.read_bytes().endswith(b"\n"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["pooled_training_universe_csv_sha256"],
            "13fb1700643f408787708190d7af752d5bd7e107d1009e6b4f6a686c0dc155ef",
        )
        self.assertEqual(
            manifest["pooling_provenance_sha256"],
            "ab908a56b5962f946c7f7fd4f2906876b1497a62f428960bfb7f71352032edca",
        )

    def test_exact_symbols_validate_and_emit_persistable_identity(self) -> None:
        identity = COHORT.validate_symbols(
            self.symbols, label="exact cohort", project_root=ROOT,
        )
        self.assertEqual(identity["schema_version"], 1)
        self.assertEqual(identity["status"], "exact_cohort_verified")
        self.assertEqual(identity["symbol_count"], 1480)
        self.assertEqual(
            identity["symbol_order_sha256"],
            COHORT.REQUIRED_SYMBOL_ORDER_SHA256,
        )
        self.assertFalse(identity["heldout_target_values_used"])
        self.assertFalse(identity["independent_final_holdout"])
        self.assertIs(COHORT.require_identity_record(
            identity, label="persisted identity",
        ), identity)
        forged = dict(identity)
        forged["origin_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            COHORT.CohortIdentityError, "bundled immutable artifact",
        ):
            COHORT.require_identity_record(forged, label="forged identity")

    def test_reordering_substitution_case_and_duplicates_fail(self) -> None:
        reordered = list(self.symbols)
        reordered[1], reordered[2] = reordered[2], reordered[1]
        substituted = list(self.symbols)
        substituted[-1] = "ZZZZZZ"
        lowercased = list(self.symbols)
        lowercased[1] = lowercased[1].lower()
        duplicated = list(self.symbols)
        duplicated[-1] = duplicated[-2]
        padded = list(self.symbols)
        padded[1] = padded[1] + " "
        short = list(self.symbols[:-1])
        cases = {
            "reordered": reordered,
            "substituted": substituted,
            "lowercased": lowercased,
            "duplicated": duplicated,
            "padded": padded,
            "short": short,
        }
        for label, symbols in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(COHORT.CohortIdentityError):
                    COHORT.validate_symbols(
                        symbols, label=label, project_root=ROOT,
                    )

    def test_csv_requires_exact_row_identity_not_only_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exact = pathlib.Path(temporary) / "exact.csv"
            wrong = pathlib.Path(temporary) / "wrong.csv"
            with exact.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=("symbol", "value"))
                writer.writeheader()
                writer.writerows(
                    {"symbol": symbol, "value": 1} for symbol in self.symbols
                )
            with wrong.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=("symbol", "value"))
                writer.writeheader()
                writer.writerows(
                    {"symbol": symbol, "value": 1}
                    for symbol in (*self.symbols[:-1], "ZZZZZZ")
                )
            identity = COHORT.validate_csv(
                exact, label="exact CSV", project_root=ROOT,
            )
            self.assertEqual(identity["symbol_count"], 1480)
            with self.assertRaises(COHORT.CohortIdentityError):
                COHORT.validate_csv(
                    wrong, label="wrong CSV", project_root=ROOT,
                )

    def source_sessions(
        self, symbols: tuple[str, ...] | list[str],
    ) -> dict[str, tuple[str, ...]]:
        canonical = tuple(symbols)
        return {
            session: canonical
            for session in COHORT.CERTIFICATION_SESSION_LABELS
        }

    def legacy_intersection(self) -> tuple[str, ...]:
        return (
            "QQQ",
            *sorted(
                set(self.symbols).union(
                    COHORT.FIXED_PRICE_GRID_EXCLUDED_SYMBOLS
                ) - {"QQQ"}
            ),
        )

    def test_prefiltered_pool_input_shape_is_accepted_and_persistable(self) -> None:
        record = COHORT.certification_pool_input_selection(
            source_sessions=self.source_sessions(self.symbols),
            excluded_symbols=(),
            final_symbols=self.symbols,
            project_root=ROOT,
        )
        self.assertEqual(
            record["mode"], COHORT.PREFILTERED_FIXED_COHORT_INPUT_MODE,
        )
        self.assertEqual(record["intersection_symbol_count"], 1480)
        self.assertEqual(record["fixed_price_grid_excluded_symbol_count"], 0)
        self.assertEqual(record["final_symbol_count"], 1480)
        self.assertTrue(record["every_source_session_is_exact_cohort"])
        self.assertEqual(
            set(record["source_session_symbol_count"].values()), {1480},
        )
        self.assertIs(
            COHORT.require_pool_input_selection_record(
                record, expected=record, label="prefiltered record",
            ),
            record,
        )

    def test_recorded_legacy_pool_input_shape_is_accepted(self) -> None:
        legacy = self.legacy_intersection()
        record = COHORT.certification_pool_input_selection(
            source_sessions=self.source_sessions(legacy),
            excluded_symbols=COHORT.FIXED_PRICE_GRID_EXCLUDED_SYMBOLS,
            final_symbols=self.symbols,
            project_root=ROOT,
        )
        self.assertEqual(
            record["mode"], COHORT.LEGACY_UNSCREENED_INPUT_MODE,
        )
        self.assertEqual(record["intersection_symbol_count"], 1509)
        self.assertEqual(record["fixed_price_grid_excluded_symbol_count"], 29)
        self.assertEqual(record["final_symbol_count"], 1480)
        self.assertFalse(record["every_source_session_is_exact_cohort"])
        self.assertEqual(
            set(record["source_session_symbol_count"].values()), {1509},
        )

    def test_mixed_or_count_only_pool_input_shapes_are_rejected(self) -> None:
        legacy = self.legacy_intersection()
        substituted_final = (
            *self.symbols[:-1], "ZZZZZZ",
        )
        wrong_legacy_exclusions = (
            *COHORT.FIXED_PRICE_GRID_EXCLUDED_SYMBOLS[:-1], "ZZZZZZ",
        )
        prefiltered_with_one_session_extra = self.source_sessions(self.symbols)
        extra_session = COHORT.CERTIFICATION_SESSION_LABELS[0]
        prefiltered_with_one_session_extra[extra_session] = (
            "QQQ",
            *sorted((*self.symbols[1:], "ZZZZZZ")),
        )
        cases = {
            "legacy-without-exclusions": {
                "source_sessions": self.source_sessions(legacy),
                "excluded_symbols": (),
                "final_symbols": self.symbols,
            },
            "prefiltered-with-legacy-exclusions": {
                "source_sessions": self.source_sessions(self.symbols),
                "excluded_symbols": (
                    COHORT.FIXED_PRICE_GRID_EXCLUDED_SYMBOLS
                ),
                "final_symbols": self.symbols,
            },
            "same-count-substituted-final": {
                "source_sessions": self.source_sessions(self.symbols),
                "excluded_symbols": (),
                "final_symbols": substituted_final,
            },
            "same-count-wrong-legacy-exclusions": {
                "source_sessions": self.source_sessions(legacy),
                "excluded_symbols": wrong_legacy_exclusions,
                "final_symbols": self.symbols,
            },
            "right-intersection-but-nonexact-source": {
                "source_sessions": prefiltered_with_one_session_extra,
                "excluded_symbols": (),
                "final_symbols": self.symbols,
            },
        }
        for label, arguments in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(COHORT.CohortIdentityError):
                    COHORT.certification_pool_input_selection(
                        **arguments, project_root=ROOT,
                    )

    def test_tampered_pool_input_record_is_rejected(self) -> None:
        expected = COHORT.certification_pool_input_selection(
            source_sessions=self.source_sessions(self.symbols),
            excluded_symbols=(),
            final_symbols=self.symbols,
            project_root=ROOT,
        )
        tampered = json.loads(json.dumps(expected))
        tampered["intersection_symbol_count"] = 1509
        with self.assertRaisesRegex(
            COHORT.CohortIdentityError, "independently reconstructed",
        ):
            COHORT.require_pool_input_selection_record(
                tampered, expected=expected, label="tampered record",
            )


if __name__ == "__main__":
    unittest.main()
