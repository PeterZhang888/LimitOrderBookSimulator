#!/usr/bin/env python3
"""Re-hash and structurally verify an applied queue-reactive data bundle.

This is the cluster-side, fail-closed counterpart to the augmentation
application.  It does not trust the provenance record alone: every merged
extractor manifest and every queue-reactive sufficient-statistics file is
located from its date/symbol identity and re-hashed before calibration starts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import stat
import sys
from collections import Counter
from collections.abc import Mapping, Sequence


REQUIRED_QUEUE_FILES = (
    "limit_buy_improvement_distribution.txt",
    "limit_sell_improvement_distribution.txt",
    "intraday_event_counts.csv",
    "queue_state_counts.csv",
    "queue_state_exposure.csv",
    "event_count_lag_moments.csv",
)
DEFAULT_DATES = (
    "2019-01-30", "2019-03-27", "2019-07-30",
    "2019-10-30", "2019-12-30", "2020-01-30",
)


class VerificationError(RuntimeError):
    """The applied empirical bundle violates its frozen provenance."""


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def direct_file(path: pathlib.Path, *, label: str) -> pathlib.Path:
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise VerificationError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise VerificationError(f"{label} must be a direct regular file: {path}")
    return path


def read_object(path: pathlib.Path, *, label: str) -> dict[str, object]:
    direct_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must contain a JSON object: {path}")
    return value


def digest_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise VerificationError(f"{label} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise VerificationError(f"{label} is not hexadecimal") from error
    return value.lower()


def improvement_counts(path: pathlib.Path) -> tuple[int, int, int]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = {
                "improvement_ticks", "improvement_price_units", "count",
            }
            if required - set(reader.fieldnames or []):
                raise VerificationError(f"invalid improvement header: {path}")
            raw = 0
            compatible = 0
            for row in reader:
                price_units = int(row["improvement_price_units"])
                ticks = float(row["improvement_ticks"])
                count = int(row["count"])
                if (
                    price_units <= 0 or count <= 0 or ticks <= 0.0
                    or abs(ticks - price_units / 100.0) > 1.0e-12
                ):
                    raise VerificationError(f"invalid improvement row: {path}")
                raw += count
                if price_units % 100 == 0:
                    compatible += count
    except (OSError, ValueError, OverflowError) as error:
        raise VerificationError(f"cannot audit improvement file {path}: {error}") from error
    return raw, compatible, raw - compatible


def verify(args: argparse.Namespace) -> dict[str, object]:
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        raise VerificationError(f"data root is not a directory: {root}")
    expected_dates = tuple(args.expected_date or DEFAULT_DATES)
    if len(expected_dates) != len(set(expected_dates)) or not expected_dates:
        raise VerificationError("expected dates must be non-empty and distinct")
    provenance_path = root / "queue_reactive_augmentation_provenance.json"
    provenance = read_object(provenance_path, label="augmentation provenance")
    records = provenance.get("records")
    expected_records = args.expected_symbols * len(expected_dates)
    if (
        provenance.get("schema_version") != 3
        or provenance.get("status") != "complete"
        or provenance.get("role") != "queue_reactive_empirical_bundle"
        or provenance.get("baseline_modified") is not False
        or not isinstance(records, list)
        or provenance.get("record_count") != expected_records
        or len(records) != expected_records
        or provenance.get("records_sha256") != sha256_json(records)
    ):
        raise VerificationError("augmentation provenance contract is incomplete")

    state_record = provenance.get("state_targets")
    if not isinstance(state_record, Mapping):
        raise VerificationError("provenance does not bind state targets")
    state_path = direct_file(
        root / "queue_reactive_state_targets.csv", label="state targets",
    )
    if sha256_file(state_path) != digest_text(
        state_record.get("sha256"), label="state-target digest",
    ):
        raise VerificationError("state-target hash mismatch")

    identities: set[tuple[str, str]] = set()
    audit_raw_total = 0
    audit_compatible_total = 0
    audit_off_grid_total = 0
    audit_affected_records = 0
    symbols_by_date: dict[str, set[str]] = {
        date: set() for date in expected_dates
    }
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise VerificationError(f"provenance record {index} is not an object")
        date = str(raw.get("trading_date", ""))
        symbol = str(raw.get("symbol", ""))
        identity = (date, symbol)
        if date not in symbols_by_date or not symbol or identity in identities:
            raise VerificationError(f"invalid or duplicate identity: {identity}")
        identities.add(identity)
        symbols_by_date[date].add(symbol)
        compact = date.replace("-", "")
        directory = root / f"itch_{compact}/empirical_data/itch_{compact}_{symbol.lower()}"
        manifest_path = directory / f"itch_manifest_{symbol.lower()}_{compact}.json"
        expected_manifest = digest_text(
            raw.get("merged_manifest_sha256"),
            label=f"merged manifest digest for {date}/{symbol}",
        )
        direct_file(manifest_path, label=f"merged manifest for {date}/{symbol}")
        if sha256_file(manifest_path) != expected_manifest:
            raise VerificationError(
                f"merged manifest hash mismatch for {date}/{symbol}"
            )
        manifest = read_object(
            manifest_path, label=f"merged manifest for {date}/{symbol}",
        )
        if (
            manifest.get("trading_date") != date
            or manifest.get("symbol") != symbol
            or not isinstance(manifest.get("queue_reactive_training_artifacts"), Mapping)
        ):
            raise VerificationError(
                f"merged manifest identity/audit block is invalid for {date}/{symbol}"
            )
        record_compatibility = raw.get("queue_reactive_runtime_compatibility")
        manifest_compatibility = manifest.get("queue_reactive_runtime_compatibility")
        if (
            not isinstance(record_compatibility, Mapping)
            or manifest_compatibility != record_compatibility
            or record_compatibility.get("schema_version") != 1
            or record_compatibility.get("status") != "passed"
            or record_compatibility.get("raw_artifacts_modified") is not False
        ):
            raise VerificationError(
                f"runtime-compatibility audit is invalid for {date}/{symbol}"
            )
        file_hashes = raw.get("queue_artifact_sha256")
        if not isinstance(file_hashes, Mapping) or set(file_hashes) != set(REQUIRED_QUEUE_FILES):
            raise VerificationError(
                f"queue-artifact file set is invalid for {date}/{symbol}"
            )
        for filename in REQUIRED_QUEUE_FILES:
            artifact = direct_file(
                directory / filename,
                label=f"{filename} for {date}/{symbol}",
            )
            expected = digest_text(
                file_hashes.get(filename),
                label=f"{filename} digest for {date}/{symbol}",
            )
            if sha256_file(artifact) != expected:
                raise VerificationError(
                    f"queue-artifact hash mismatch: {date}/{symbol}/{filename}"
                )
        buy_counts = improvement_counts(
            directory / "limit_buy_improvement_distribution.txt"
        )
        sell_counts = improvement_counts(
            directory / "limit_sell_improvement_distribution.txt"
        )
        observed_counts = tuple(
            buy_counts[index] + sell_counts[index] for index in range(3)
        )
        recorded_counts = (
            record_compatibility.get("raw_exact_inside_spread_mark_count"),
            record_compatibility.get("runtime_compatible_mark_count"),
            record_compatibility.get("excluded_off_grid_mark_count"),
        )
        if recorded_counts != observed_counts:
            raise VerificationError(
                f"runtime-compatibility counts differ for {date}/{symbol}"
            )
        audit_raw_total += observed_counts[0]
        audit_compatible_total += observed_counts[1]
        audit_off_grid_total += observed_counts[2]
        audit_affected_records += int(observed_counts[2] > 0)

    reference_symbols: set[str] | None = None
    for date in expected_dates:
        symbols = symbols_by_date[date]
        if len(symbols) != args.expected_symbols:
            raise VerificationError(
                f"{date} has {len(symbols)} symbols, expected {args.expected_symbols}"
            )
        if reference_symbols is None:
            reference_symbols = symbols
        elif symbols != reference_symbols:
            raise VerificationError("the six dates do not share one exact symbol cohort")

    try:
        with state_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            state_symbols = [str(row.get("symbol", "")) for row in reader]
    except OSError as error:
        raise VerificationError(f"cannot read state targets: {error}") from error
    counts = Counter(state_symbols)
    if (
        reference_symbols is None
        or set(counts) != reference_symbols
        or any(count != 1 for count in counts.values())
    ):
        raise VerificationError("state-target symbol cohort is not the common cohort")

    expected_summary = {
        "status": "passed",
        "raw_exact_inside_spread_mark_count": audit_raw_total,
        "runtime_compatible_mark_count": audit_compatible_total,
        "excluded_off_grid_mark_count": audit_off_grid_total,
        "records_with_off_grid_marks": audit_affected_records,
        "raw_artifacts_modified": False,
    }
    if provenance.get("runtime_compatibility_summary") != expected_summary:
        raise VerificationError("runtime-compatibility summary is inconsistent")

    return {
        "status": "passed",
        "schema_version": 1,
        "data_root": str(root),
        "provenance": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "date_count": len(expected_dates),
        "symbol_count": args.expected_symbols,
        "record_count": len(records),
        "verified_file_count": 1 + len(records) * (1 + len(REQUIRED_QUEUE_FILES)),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", type=pathlib.Path, required=True)
    result.add_argument("--expected-symbols", type=int, default=1480)
    result.add_argument(
        "--expected-date", action="append",
        help="repeat for each expected date; defaults to the frozen six-date cohort",
    )
    result.add_argument("--output", type=pathlib.Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        if args.expected_symbols <= 0:
            raise VerificationError("expected-symbols must be positive")
        report = verify(args)
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
    except (VerificationError, OSError, ValueError) as error:
        print(f"queue-reactive empirical verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
