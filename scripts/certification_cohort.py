#!/usr/bin/env python3
"""Fail-closed identity contract for the fixed 1,480-symbol cohort.

The development-validation cohort was selected once from the fixed balanced
panel.  This module deliberately distinguishes *which* symbols are used from
the weaker statement that a configuration merely contains 1,480 rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
from typing import Iterable, Mapping


REQUIRED_SYMBOL_COUNT = 1_480
REQUIRED_SYMBOL_ORDER_SHA256 = (
    "2f57f37762772d9523fb9916fe2376a9578e337d20971fe39aa44d578f5691d3"
)
POOLED_TRAINING_CSV_SHA256 = (
    "13fb1700643f408787708190d7af752d5bd7e107d1009e6b4f6a686c0dc155ef"
)
POOLING_PROVENANCE_SHA256 = (
    "ab908a56b5962f946c7f7fd4f2906876b1497a62f428960bfb7f71352032edca"
)
ORIGIN_MANIFEST_SHA256 = (
    "d0c881e8b01c89e4bc1ee99d0766484fb1031f434a01c26d492b074ca089a4de"
)
COHORT_RELATIVE_PATH = pathlib.Path("config/certification_symbols_1480.txt")
ORIGIN_MANIFEST_RELATIVE_PATH = pathlib.Path(
    "config/certification_symbols_1480_origin.json"
)

# The fixed balanced panel was obtained by applying the fixed one-cent opening
# price-domain screen to a 1,509-symbol six-session intersection.  Fresh
# extraction may instead predeclare the already-screened 1,480-symbol panel on
# every session.  Certification admits exactly these two input shapes and no
# count-only approximation of either one.
LEGACY_UNSCREENED_INPUT_MODE = "legacy_unscreened_1509_to_1480"
PREFILTERED_FIXED_COHORT_INPUT_MODE = (
    "prefiltered_fixed_cohort_1480_to_1480"
)
CERTIFICATION_SESSION_LABELS = (
    "2019-01-30", "2019-03-27", "2019-07-30", "2019-10-30",
    "2019-12-30", "2020-01-30",
)
FIXED_PRICE_GRID_EXCLUDED_SYMBOLS = (
    "ABUS", "ACHV", "ACST", "ADRO", "AEZS", "ALOT", "ASPU", "ASRT",
    "AYTU", "BLCM", "BSQR", "CHCI", "CMLS", "CSSE", "EAST", "EXFO",
    "FORD", "GLBS", "HSDT", "INAP", "INFI", "INVE", "MBRX", "MFNC",
    "OSS", "PHUN", "STAF", "SVRA", "TESS",
)


class CohortIdentityError(ValueError):
    """Raised when an artifact does not contain the frozen cohort exactly."""


def canonical_symbols(symbols: Iterable[object], *, label: str) -> tuple[str, ...]:
    """Require exact uppercase, unique, QQQ-first lexicographic symbols."""
    normalized: list[str] = []
    for index, raw in enumerate(symbols):
        raw_text = str(raw)
        value = raw_text.strip()
        symbol = value.upper()
        if not value or raw_text != value or value != symbol:
            raise CohortIdentityError(
                f"{label} symbol {index} is not exact non-empty uppercase text: "
                f"{raw!r}"
            )
        try:
            symbol.encode("ascii")
        except UnicodeEncodeError as error:
            raise CohortIdentityError(
                f"{label} symbol {index} is not ASCII: {symbol!r}"
            ) from error
        normalized.append(symbol)
    if len(normalized) != len(set(normalized)):
        raise CohortIdentityError(f"{label} contains duplicate symbols")
    expected_order = (
        ("QQQ",) + tuple(sorted(symbol for symbol in normalized if symbol != "QQQ"))
        if "QQQ" in normalized else tuple(sorted(normalized))
    )
    observed = tuple(normalized)
    if observed != expected_order:
        raise CohortIdentityError(
            f"{label} is not in canonical QQQ-first then lexicographic order"
        )
    return observed


def canonical_bytes(symbols: Iterable[object], *, label: str) -> bytes:
    values = canonical_symbols(symbols, label=label)
    return ("\n".join(values) + "\n").encode("utf-8")


def symbol_order_sha256(symbols: Iterable[object], *, label: str) -> str:
    return hashlib.sha256(canonical_bytes(symbols, label=label)).hexdigest()


def _sequence_sha256(symbols: Iterable[object], *, label: str) -> str:
    """Hash a canonical sequence, using the empty byte string for no symbols."""
    values = canonical_symbols(symbols, label=label)
    rendered = b"" if not values else ("\n".join(values) + "\n").encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def project_paths(project_root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    root = project_root.expanduser().resolve()
    return root / COHORT_RELATIVE_PATH, root / ORIGIN_MANIFEST_RELATIVE_PATH


def load_required_symbols(project_root: pathlib.Path) -> tuple[str, ...]:
    cohort_path, manifest_path = project_paths(project_root)
    if not cohort_path.is_file():
        raise CohortIdentityError(f"bundled certification cohort is missing: {cohort_path}")
    raw = cohort_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CohortIdentityError("bundled certification cohort is not UTF-8") from error
    symbols = canonical_symbols(text.splitlines(), label="bundled certification cohort")
    if raw != ("\n".join(symbols) + "\n").encode("utf-8"):
        raise CohortIdentityError(
            "bundled certification cohort must use one symbol per line and one final LF"
        )
    observed_hash = hashlib.sha256(raw).hexdigest()
    if len(symbols) != REQUIRED_SYMBOL_COUNT or observed_hash != REQUIRED_SYMBOL_ORDER_SHA256:
        raise CohortIdentityError(
            "bundled certification cohort disagrees with its immutable count or SHA-256"
        )
    if not manifest_path.is_file():
        raise CohortIdentityError(
            f"bundled certification cohort origin manifest is missing: {manifest_path}"
        )
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CohortIdentityError(
            f"cannot parse certification cohort origin manifest: {error}"
        ) from error
    expected = {
        "schema_version": 1,
        "artifact_role": "certification_cohort_origin_manifest",
        "cohort_file": COHORT_RELATIVE_PATH.as_posix(),
        "cohort_symbol_count": REQUIRED_SYMBOL_COUNT,
        "cohort_symbol_order_sha256": REQUIRED_SYMBOL_ORDER_SHA256,
        "canonicalization": "uppercase_UTF-8_QQQ_first_then_lexicographic_final_LF",
        "selection_role": "development_validation_balanced_panel",
        "heldout_availability_conditioned": True,
        "heldout_target_values_used": False,
        "independent_final_holdout": False,
        "original_intersection_symbol_count": 1_509,
        "fixed_price_grid_excluded_symbol_count": 29,
        "final_symbol_count": REQUIRED_SYMBOL_COUNT,
        "pooled_training_universe_csv_sha256": POOLED_TRAINING_CSV_SHA256,
        "pooling_provenance_sha256": POOLING_PROVENANCE_SHA256,
    }
    if manifest != expected:
        raise CohortIdentityError(
            "bundled certification cohort origin manifest is not canonical"
        )
    if hashlib.sha256(manifest_bytes).hexdigest() != ORIGIN_MANIFEST_SHA256:
        raise CohortIdentityError(
            "bundled certification cohort origin manifest has the wrong SHA-256"
        )
    return symbols


def validate_symbols(
    symbols: Iterable[object], *, label: str, project_root: pathlib.Path,
) -> dict[str, object]:
    observed = canonical_symbols(symbols, label=label)
    required = load_required_symbols(project_root)
    observed_hash = symbol_order_sha256(observed, label=label)
    if observed != required or observed_hash != REQUIRED_SYMBOL_ORDER_SHA256:
        raise CohortIdentityError(
            f"{label} is not the immutable {REQUIRED_SYMBOL_COUNT}-symbol cohort; "
            f"observed count={len(observed)} sha256={observed_hash}"
        )
    cohort_path, manifest_path = project_paths(project_root)
    return {
        "schema_version": 1,
        "status": "exact_cohort_verified",
        "symbol_count": len(observed),
        "symbol_order_sha256": observed_hash,
        "canonical_order": "QQQ_first_then_lexicographic",
        "cohort_file": COHORT_RELATIVE_PATH.as_posix(),
        "cohort_file_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        "origin_manifest": ORIGIN_MANIFEST_RELATIVE_PATH.as_posix(),
        "origin_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "selection_role": "development_validation_balanced_panel",
        "heldout_availability_conditioned": True,
        "heldout_target_values_used": False,
        "independent_final_holdout": False,
    }


def symbols_from_csv(path: pathlib.Path, *, label: str) -> tuple[str, ...]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or "symbol" not in reader.fieldnames:
                raise CohortIdentityError(f"{label} lacks a symbol column")
            rows = list(reader)
    except OSError as error:
        raise CohortIdentityError(f"cannot read {label}: {error}") from error
    if not rows:
        raise CohortIdentityError(f"{label} contains no rows")
    return canonical_symbols(
        (row.get("symbol", "") for row in rows), label=label,
    )


def validate_csv(
    path: pathlib.Path, *, label: str, project_root: pathlib.Path,
) -> dict[str, object]:
    return validate_symbols(
        symbols_from_csv(path, label=label), label=label, project_root=project_root,
    )


def require_identity_record(value: object, *, label: str) -> Mapping[str, object]:
    """Validate the immutable semantic fields of a persisted identity record."""
    if not isinstance(value, Mapping):
        raise CohortIdentityError(f"{label} is not an object")
    expected = {
        "schema_version": 1,
        "status": "exact_cohort_verified",
        "symbol_count": REQUIRED_SYMBOL_COUNT,
        "symbol_order_sha256": REQUIRED_SYMBOL_ORDER_SHA256,
        "canonical_order": "QQQ_first_then_lexicographic",
        "cohort_file": COHORT_RELATIVE_PATH.as_posix(),
        "selection_role": "development_validation_balanced_panel",
        "heldout_availability_conditioned": True,
        "heldout_target_values_used": False,
        "independent_final_holdout": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise CohortIdentityError(f"{label}.{key} is not canonical")
    expected_artifact_hashes = {
        "cohort_file_sha256": REQUIRED_SYMBOL_ORDER_SHA256,
        "origin_manifest_sha256": ORIGIN_MANIFEST_SHA256,
    }
    for key, expected_hash in expected_artifact_hashes.items():
        if value.get(key) != expected_hash:
            raise CohortIdentityError(
                f"{label}.{key} does not identify the bundled immutable artifact"
            )
    return value


def certification_pool_input_selection(
    *,
    source_sessions: Mapping[str, Iterable[object]],
    excluded_symbols: Iterable[object],
    final_symbols: Iterable[object],
    project_root: pathlib.Path,
) -> dict[str, object]:
    """Validate and identify one of the two admissible pool input shapes.

    The function derives the intersection itself from all six source-session
    symbol sequences.  Consequently a caller cannot make an inadmissible
    universe appear valid merely by supplying the expected row counts in JSON.
    """
    if tuple(sorted(source_sessions)) != tuple(sorted(CERTIFICATION_SESSION_LABELS)):
        raise CohortIdentityError(
            "certification pool sources must cover exactly the five training "
            "sessions and 2020-01-30"
        )
    normalized_sources = {
        session: canonical_symbols(
            symbols, label=f"certification pool source {session}",
        )
        for session, symbols in source_sessions.items()
    }
    required = load_required_symbols(project_root)
    final = canonical_symbols(final_symbols, label="certification pool final cohort")
    validate_symbols(
        final, label="certification pool final cohort", project_root=project_root,
    )
    excluded = canonical_symbols(
        excluded_symbols, label="certification pool fixed-grid exclusions",
    )

    common = set(next(iter(normalized_sources.values())))
    for symbols in normalized_sources.values():
        common.intersection_update(symbols)
    intersection = canonical_symbols(
        ("QQQ", *sorted(symbol for symbol in common if symbol != "QQQ")),
        label="certification pool source intersection",
    )
    excluded_set = set(excluded)
    if not excluded_set.issubset(common):
        raise CohortIdentityError(
            "certification fixed-grid exclusions are not a subset of the source "
            "intersection"
        )
    derived_final = canonical_symbols(
        (
            "QQQ",
            *sorted(
                symbol for symbol in common
                if symbol != "QQQ" and symbol not in excluded_set
            ),
        ),
        label="certification pool screened source intersection",
    )
    if derived_final != final:
        raise CohortIdentityError(
            "certification source intersection minus fixed-grid exclusions does "
            "not equal the immutable final cohort"
        )

    legacy_intersection = canonical_symbols(
        (
            "QQQ",
            *sorted(
                set(required).union(FIXED_PRICE_GRID_EXCLUDED_SYMBOLS)
                - {"QQQ"}
            ),
        ),
        label="recorded pre-screen intersection",
    )
    if (intersection == legacy_intersection
            and excluded == FIXED_PRICE_GRID_EXCLUDED_SYMBOLS):
        mode = LEGACY_UNSCREENED_INPUT_MODE
        every_source_is_exact_cohort = False
    elif intersection == required and not excluded:
        # The fresh fixed-universe workflow promises more than an intersection:
        # every daily source configuration is itself exactly the frozen panel.
        nonexact = [
            session for session, symbols in normalized_sources.items()
            if symbols != required
        ]
        if nonexact:
            raise CohortIdentityError(
                "prefiltered certification input has the right intersection but "
                "one or more source sessions are not the exact cohort: "
                + ", ".join(sorted(nonexact))
            )
        mode = PREFILTERED_FIXED_COHORT_INPUT_MODE
        every_source_is_exact_cohort = True
    else:
        raise CohortIdentityError(
            "unsupported certification pool input shape: "
            f"intersection={len(intersection)} exclusions={len(excluded)}; "
            "expected the recorded 1509-to-1480 fixed-grid screen or six exact "
            "prefiltered 1480-symbol sessions"
        )

    return {
        "schema_version": 1,
        "status": "exact_certification_pool_input_verified",
        "mode": mode,
        "source_session_count": len(normalized_sources),
        "source_sessions": list(CERTIFICATION_SESSION_LABELS),
        "source_session_symbol_count": {
            session: len(normalized_sources[session])
            for session in CERTIFICATION_SESSION_LABELS
        },
        "source_session_symbol_order_sha256": {
            session: _sequence_sha256(
                normalized_sources[session],
                label=f"certification pool source {session}",
            )
            for session in CERTIFICATION_SESSION_LABELS
        },
        "every_source_session_is_exact_cohort": every_source_is_exact_cohort,
        "intersection_symbol_count": len(intersection),
        "intersection_symbol_order_sha256": _sequence_sha256(
            intersection, label="certification pool source intersection",
        ),
        "fixed_price_grid_excluded_symbol_count": len(excluded),
        "fixed_price_grid_excluded_symbol_order_sha256": _sequence_sha256(
            excluded, label="certification pool fixed-grid exclusions",
        ),
        "final_symbol_count": len(final),
        "final_symbol_order_sha256": _sequence_sha256(
            final, label="certification pool final cohort",
        ),
    }


def require_pool_input_selection_record(
    value: object, *, expected: Mapping[str, object], label: str,
) -> Mapping[str, object]:
    """Require byte-for-byte semantic equality with an independently rebuilt record."""
    if not isinstance(value, Mapping):
        raise CohortIdentityError(f"{label} is not an object")
    if dict(value) != dict(expected):
        raise CohortIdentityError(
            f"{label} does not match the independently reconstructed source shape"
        )
    return value
