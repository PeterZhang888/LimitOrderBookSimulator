#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Merge audited queue-reactive sidecars into a new compact-data root.

The baseline data root is never edited.  Files are hard-linked into a sibling
temporary tree, the new sufficient statistics are copied, and each legacy
extractor manifest receives the certified queue-reactive audit block.  When
the retained extractor manifest proves exact prefix snapshot coverage that an
older compact manifest omitted, the application also restores the integer
valid/invalid prefix counts.  Existing empirical targets are never replaced.
The completed tree is renamed into place atomically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Mapping, Sequence


REQUIRED_QUEUE_FILES = (
    "limit_buy_improvement_distribution.txt",
    "limit_sell_improvement_distribution.txt",
    "intraday_event_counts.csv",
    "queue_state_counts.csv",
    "queue_state_exposure.csv",
    "event_count_lag_moments.csv",
)


class ApplicationError(RuntimeError):
    """The empirical augmentation cannot be applied without mutation risk."""


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path: pathlib.Path, value: object) -> None:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True,
    )
    temporary = pathlib.Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_json(path: pathlib.Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ApplicationError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ApplicationError(f"JSON object required: {path}")
    return value


def equal_json_value(left: object, right: object) -> bool:
    """Compare extractor estimands without accepting string coercion."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        left_number = float(left)
        right_number = float(right)
        return (
            math.isfinite(left_number)
            and math.isfinite(right_number)
            and math.isclose(
                left_number, right_number, rel_tol=1.0e-12, abs_tol=1.0e-15,
            )
        )
    return left == right


def recover_prefix_snapshot_accounting(
    legacy: Mapping[str, object],
    source: Mapping[str, object],
    *,
    source_manifest_path: pathlib.Path,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Restore exact prefix counts from the retained extractor manifest.

    The queue-reactive extraction retained a byte-bound copy of its source
    manifest.  Earlier compact bundles omitted prefix coverage even though
    all other prefix targets were preserved.  A count is restored only when
    identities, source-file provenance and every pre-existing target/scale in
    that prefix agree.  This deliberately does not import full-session target
    values or alter any target CSV.
    """
    source_windows = source.get("market_target_windows")
    if not isinstance(source_windows, Mapping):
        return dict(legacy), None
    legacy_windows = legacy.get("market_target_windows")
    if not isinstance(legacy_windows, Mapping):
        raise ApplicationError(
            "source extractor records prefix targets but legacy manifest does not"
        )

    for field in (
        "trading_date", "symbol", "input_sha256", "input_size_bytes",
        "session_start", "session_end", "snapshot_interval_ms",
        "valid_snapshots", "invalid_snapshots",
        "distribution_observation_counts", "placement_counts",
    ):
        source_value = source.get(field)
        legacy_value = legacy.get(field)
        if source_value is not None and legacy_value is not None:
            if source_value != legacy_value:
                raise ApplicationError(
                    f"source/legacy extractor provenance differs for {field}"
                )

    merged = dict(legacy)
    merged_windows: dict[str, object] = dict(legacy_windows)
    recovered: dict[str, object] = {}
    for raw_horizon, raw_source_record in sorted(source_windows.items()):
        horizon = str(raw_horizon)
        if not isinstance(raw_source_record, Mapping):
            raise ApplicationError(
                f"source extractor prefix record is malformed for {horizon}"
            )
        raw_legacy_record = legacy_windows.get(horizon)
        if not isinstance(raw_legacy_record, Mapping):
            continue
        for field in ("duration_seconds", "observations", "file"):
            if raw_source_record.get(field) != raw_legacy_record.get(field):
                raise ApplicationError(
                    f"source/legacy prefix provenance differs for {horizon}/{field}"
                )
        observations = raw_source_record.get("observations")
        if (
            isinstance(observations, bool)
            or not isinstance(observations, int)
            or observations <= 0
        ):
            raise ApplicationError(
                f"source extractor has invalid prefix observations for {horizon}"
            )

        source_values = raw_source_record.get("values")
        source_scales = raw_source_record.get("scales")
        legacy_values = raw_legacy_record.get("values")
        legacy_scales = raw_legacy_record.get("scales")
        if not all(isinstance(value, Mapping) for value in (
            source_values, source_scales, legacy_values, legacy_scales,
        )):
            raise ApplicationError(
                f"source/legacy prefix targets are malformed for {horizon}"
            )
        assert isinstance(source_values, Mapping)
        assert isinstance(source_scales, Mapping)
        assert isinstance(legacy_values, Mapping)
        assert isinstance(legacy_scales, Mapping)
        coverage_metric = "two_sided_sample_fraction"
        for role, legacy_map, source_map in (
            ("value", legacy_values, source_values),
            ("scale", legacy_scales, source_scales),
        ):
            for metric, legacy_value in legacy_map.items():
                if metric == coverage_metric:
                    continue
                if metric not in source_map or not equal_json_value(
                    legacy_value, source_map[metric],
                ):
                    raise ApplicationError(
                        f"source/legacy prefix {role} differs for "
                        f"{horizon}/{metric}"
                    )

        coverage_value = source_values.get(coverage_metric)
        coverage_scale = source_scales.get(coverage_metric)
        if coverage_value is None and coverage_scale is None:
            continue
        if coverage_value is None or coverage_scale is None:
            raise ApplicationError(
                f"source extractor has incomplete prefix coverage for {horizon}"
            )
        if isinstance(coverage_value, bool) or not isinstance(
            coverage_value, (int, float),
        ):
            raise ApplicationError(
                f"source extractor has invalid prefix coverage for {horizon}"
            )
        coverage = float(coverage_value)
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ApplicationError(
                f"source extractor has invalid prefix coverage for {horizon}"
            )
        raw_valid = coverage * observations
        valid = int(round(raw_valid))
        if not math.isclose(raw_valid, valid, rel_tol=0.0, abs_tol=1.0e-8):
            raise ApplicationError(
                f"source prefix coverage is not an integer count for {horizon}"
            )
        invalid = observations - valid
        existing_valid = raw_legacy_record.get("valid_snapshots")
        existing_invalid = raw_legacy_record.get("invalid_snapshots")
        if existing_valid is not None or existing_invalid is not None:
            if existing_valid != valid or existing_invalid != invalid:
                raise ApplicationError(
                    f"legacy prefix counts disagree with source for {horizon}"
                )
        merged_record = dict(raw_legacy_record)
        merged_record["valid_snapshots"] = valid
        merged_record["invalid_snapshots"] = invalid
        merged_windows[horizon] = merged_record
        recovered[horizon] = {
            "observations": observations,
            "valid_snapshots": valid,
            "invalid_snapshots": invalid,
            "source_two_sided_sample_fraction": coverage,
            "source_empirical_scale": coverage_scale,
        }

    if not recovered:
        return merged, None
    audit = {
        "schema_version": 1,
        "status": "exact_counts_recovered",
        "method": (
            "integer counts reconstructed from the retained extractor's "
            "two-sided prefix fraction after equality checks on all existing "
            "prefix targets and scales"
        ),
        "source_manifest_relative_name": source_manifest_path.name,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "existing_empirical_targets_replaced": False,
        "windows": recovered,
    }
    merged["market_target_windows"] = merged_windows
    merged["prefix_snapshot_accounting_recovery"] = audit
    return merged, audit


@dataclass(frozen=True)
class ImprovementAudit:
    raw_count: int
    runtime_compatible_count: int
    excluded_off_grid_count: int


def improvement_audit(path: pathlib.Path) -> ImprovementAudit:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = {
                "improvement_ticks", "improvement_price_units", "count",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ApplicationError(
                    f"improvement file lacks required columns {sorted(missing)}: {path}"
                )
            raw_count = 0
            compatible_count = 0
            for row in reader:
                count = int(row["count"])
                price_units = int(row["improvement_price_units"])
                ticks = float(row["improvement_ticks"])
                if count <= 0 or price_units <= 0:
                    raise ApplicationError(
                        f"improvement distance and count must be positive: {path}"
                    )
                if not ticks > 0.0 or abs(ticks - price_units / 100.0) > 1.0e-12:
                    raise ApplicationError(
                        f"inconsistent improvement units/ticks in {path}"
                    )
                raw_count += count
                if price_units % 100 == 0:
                    compatible_count += count
            return ImprovementAudit(
                raw_count=raw_count,
                runtime_compatible_count=compatible_count,
                excluded_off_grid_count=raw_count - compatible_count,
            )
    except (OSError, ValueError, OverflowError) as error:
        raise ApplicationError(f"cannot read improvement file {path}: {error}") from error


def verify_augmentation(
    root: pathlib.Path, manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("role")
        != "queue_reactive_sufficient_statistics_augmentation"
        or manifest.get("legacy_empirical_bundle_modified") is not False
    ):
        raise ApplicationError("unsupported or incomplete augmentation manifest")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ApplicationError("augmentation manifest has no records")
    if manifest.get("record_count") != len(records):
        raise ApplicationError("augmentation record count is inconsistent")
    if manifest.get("records_sha256") != sha256_json(records):
        raise ApplicationError("augmentation record digest is invalid")
    identities: set[tuple[str, str]] = set()
    normalized: list[dict[str, object]] = []
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise ApplicationError("invalid augmentation record")
        trading_date = str(raw_record.get("trading_date", ""))
        symbol = str(raw_record.get("symbol", ""))
        identity = (trading_date, symbol)
        if not trading_date or not symbol or identity in identities:
            raise ApplicationError(f"invalid or duplicate record identity: {identity}")
        identities.add(identity)
        relative = pathlib.PurePosixPath(str(raw_record.get("relative_directory", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ApplicationError(f"unsafe augmentation path: {relative}")
        expected_prefix = pathlib.PurePosixPath(
            f"itch_{trading_date.replace('-', '')}/empirical_data/"
            f"itch_{trading_date.replace('-', '')}_{symbol.lower()}"
        )
        if relative != expected_prefix:
            raise ApplicationError(f"record path/identity mismatch: {relative}")
        files = raw_record.get("files")
        if not isinstance(files, list):
            raise ApplicationError(f"record files are missing: {identity}")
        by_name: dict[str, dict[str, object]] = {}
        for item in files:
            if not isinstance(item, dict):
                raise ApplicationError(f"invalid file record: {identity}")
            name = str(item.get("relative_name", ""))
            if pathlib.PurePath(name).name != name or name in by_name:
                raise ApplicationError(f"unsafe or duplicate file name: {name}")
            source = root / pathlib.Path(*relative.parts) / name
            if not source.is_file() or sha256_file(source) != item.get("sha256"):
                raise ApplicationError(f"augmentation file missing or mismatched: {source}")
            by_name[name] = item
        required = {
            *REQUIRED_QUEUE_FILES,
            "queue_reactive_training_artifacts.json",
            "source_extractor_manifest.json",
        }
        if set(by_name) != required:
            raise ApplicationError(f"augmentation file set differs for {identity}")
        normalized.append({
            "trading_date": trading_date,
            "symbol": symbol,
            "relative_directory": relative,
            "files": by_name,
        })
    return normalized


def hardlink(source: str, destination: str) -> str:
    os.link(source, destination)
    return destination


def apply(args: argparse.Namespace) -> dict[str, object]:
    baseline = args.baseline_root.expanduser().resolve()
    augmentation = args.augmentation_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if not baseline.is_dir() or not augmentation.is_dir():
        raise ApplicationError("baseline and augmentation roots must be directories")
    if output.exists():
        raise ApplicationError(f"output root already exists: {output}")
    manifest_path = augmentation / "queue_reactive_augmentation_manifest.json"
    manifest = read_json(manifest_path)
    records = verify_augmentation(augmentation, manifest)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise ApplicationError(f"temporary output already exists: {temporary}")
    producer_script = pathlib.Path(__file__).resolve()
    applied: list[dict[str, object]] = []
    try:
        copy_function = shutil.copy2 if args.copy_files else hardlink
        shutil.copytree(baseline, temporary, copy_function=copy_function)
        for index, record in enumerate(records, start=1):
            trading_date = str(record["trading_date"])
            symbol = str(record["symbol"])
            relative = record["relative_directory"]
            if not isinstance(relative, pathlib.PurePosixPath):
                raise ApplicationError("internal relative-path type mismatch")
            source_dir = augmentation / pathlib.Path(*relative.parts)
            target_dir = temporary / pathlib.Path(*relative.parts)
            if not target_dir.is_dir():
                raise ApplicationError(
                    f"baseline lacks symbol directory for {trading_date}/{symbol}"
                )
            compact = trading_date.replace("-", "")
            legacy_manifest_path = (
                target_dir / f"itch_manifest_{symbol.lower()}_{compact}.json"
            )
            legacy_manifest = read_json(legacy_manifest_path)
            if (
                legacy_manifest.get("trading_date") != trading_date
                or legacy_manifest.get("symbol") != symbol
            ):
                raise ApplicationError(
                    f"legacy manifest identity mismatch for {trading_date}/{symbol}"
                )
            if "queue_reactive_training_artifacts" in legacy_manifest:
                raise ApplicationError(
                    f"baseline is already queue-reactive: {trading_date}/{symbol}"
                )
            sidecar = read_json(
                source_dir / "queue_reactive_training_artifacts.json"
            )
            source_manifest_path = source_dir / "source_extractor_manifest.json"
            source_manifest = read_json(source_manifest_path)
            if (
                sidecar.get("trading_date") != trading_date
                or sidecar.get("symbol") != symbol
            ):
                raise ApplicationError(
                    f"sidecar identity mismatch for {trading_date}/{symbol}"
                )
            block = sidecar.get("queue_reactive_training_artifacts")
            if not isinstance(block, dict):
                raise ApplicationError(f"sidecar audit block is absent for {symbol}")
            conservation = block.get("event_count_conservation")
            if not isinstance(conservation, dict):
                raise ApplicationError(f"sidecar conservation audit is absent for {symbol}")
            observed_counts = conservation.get("by_event_type")
            legacy_counts = legacy_manifest.get("distribution_observation_counts")
            if observed_counts != legacy_counts:
                raise ApplicationError(
                    f"new/legacy event counts differ for {trading_date}/{symbol}"
                )
            if sidecar.get("legacy_distribution_observation_counts") != legacy_counts:
                raise ApplicationError(
                    f"sidecar legacy-count record differs for {trading_date}/{symbol}"
                )
            side_audits = {
                side: improvement_audit(
                    source_dir / f"limit_{side}_improvement_distribution.txt"
                )
                for side in ("buy", "sell")
            }
            raw_inside = sum(item.raw_count for item in side_audits.values())
            compatible_inside = sum(
                item.runtime_compatible_count for item in side_audits.values()
            )
            off_grid_inside = sum(
                item.excluded_off_grid_count for item in side_audits.values()
            )
            placement = legacy_manifest.get("placement_counts")
            expected_inside = (
                placement.get("inside_spread_limit_orders")
                if isinstance(placement, dict) else None
            )
            if not isinstance(expected_inside, int) or expected_inside < 0:
                raise ApplicationError(
                    f"legacy inside-spread count is invalid for {trading_date}/{symbol}"
                )
            if off_grid_inside == 0 and raw_inside != expected_inside:
                raise ApplicationError(
                    f"inside-spread count differs for {trading_date}/{symbol}: "
                    f"{raw_inside} != {expected_inside}"
                )
            if off_grid_inside > 0 and not (
                compatible_inside <= expected_inside <= raw_inside
            ):
                raise ApplicationError(
                    f"inside-spread estimands are incompatible for "
                    f"{trading_date}/{symbol}: runtime-compatible="
                    f"{compatible_inside}, legacy={expected_inside}, raw={raw_inside}"
                )
            runtime_compatibility = {
                "schema_version": 1,
                "status": "passed",
                "raw_exact_inside_spread_mark_count": raw_inside,
                "legacy_one_cent_eligibility_count": expected_inside,
                "runtime_compatible_mark_count": compatible_inside,
                "excluded_off_grid_mark_count": off_grid_inside,
                "runtime_price_grid_units": 100,
                "projection": (
                    "retain positive exact ITCH price improvements divisible by "
                    "100 price units; preserve all raw rows in source artifacts"
                ),
                "raw_artifacts_modified": False,
                "sides": {
                    side: {
                        "raw_exact_count": audit.raw_count,
                        "runtime_compatible_count": audit.runtime_compatible_count,
                        "excluded_off_grid_count": audit.excluded_off_grid_count,
                    }
                    for side, audit in sorted(side_audits.items())
                },
            }
            legacy_hash = sha256_file(legacy_manifest_path)
            for filename in REQUIRED_QUEUE_FILES:
                destination = target_dir / filename
                if destination.exists():
                    raise ApplicationError(
                        f"baseline unexpectedly contains queue artifact: {destination}"
                    )
                shutil.copy2(source_dir / filename, destination)
            merged, prefix_accounting = recover_prefix_snapshot_accounting(
                legacy_manifest,
                source_manifest,
                source_manifest_path=source_manifest_path,
            )
            merged["queue_reactive_training_artifacts"] = block
            merged["queue_reactive_runtime_compatibility"] = runtime_compatibility
            atomic_json(legacy_manifest_path, merged)
            applied.append({
                "trading_date": trading_date,
                "symbol": symbol,
                "legacy_manifest_sha256": legacy_hash,
                "merged_manifest_sha256": sha256_file(legacy_manifest_path),
                "queue_artifact_sha256": {
                    filename: sha256_file(target_dir / filename)
                    for filename in REQUIRED_QUEUE_FILES
                },
                "queue_reactive_runtime_compatibility": runtime_compatibility,
                "prefix_snapshot_accounting_recovery": prefix_accounting,
            })
            if index % 500 == 0:
                print(f"applied {index}/{len(records)} records", flush=True)
        shutil.copy2(
            augmentation / "queue_reactive_state_targets.csv",
            temporary / "queue_reactive_state_targets.csv",
        )
        state_targets_path = temporary / "queue_reactive_state_targets.csv"
        raw_inside_total = sum(
            int(record["queue_reactive_runtime_compatibility"]
                ["raw_exact_inside_spread_mark_count"])
            for record in applied
        )
        compatible_inside_total = sum(
            int(record["queue_reactive_runtime_compatibility"]
                ["runtime_compatible_mark_count"])
            for record in applied
        )
        off_grid_inside_total = sum(
            int(record["queue_reactive_runtime_compatibility"]
                ["excluded_off_grid_mark_count"])
            for record in applied
        )
        provenance = {
            "schema_version": 3,
            "status": "complete",
            "role": "queue_reactive_empirical_bundle",
            "baseline_root": str(baseline),
            "baseline_modified": False,
            "copy_mode": "physical_copy" if args.copy_files else "hardlink_copy_on_write",
            "augmentation_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "application_script": {
                "path": str(producer_script),
                "sha256": sha256_file(producer_script),
            },
            "state_targets": {
                "relative_path": "queue_reactive_state_targets.csv",
                "sha256": sha256_file(state_targets_path),
            },
            "record_count": len(applied),
            "runtime_compatibility_summary": {
                "status": "passed",
                "raw_exact_inside_spread_mark_count": raw_inside_total,
                "runtime_compatible_mark_count": compatible_inside_total,
                "excluded_off_grid_mark_count": off_grid_inside_total,
                "records_with_off_grid_marks": sum(
                    int(record["queue_reactive_runtime_compatibility"]
                        ["excluded_off_grid_mark_count"]) > 0
                    for record in applied
                ),
                "raw_artifacts_modified": False,
            },
            "prefix_snapshot_accounting_recovery_summary": {
                "status": "complete",
                "records_with_recovered_prefix_counts": sum(
                    record["prefix_snapshot_accounting_recovery"] is not None
                    for record in applied
                ),
                "existing_empirical_targets_replaced": False,
            },
            "records_sha256": sha256_json(applied),
            "records": applied,
        }
        atomic_json(
            temporary / "queue_reactive_augmentation_provenance.json",
            provenance,
        )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "status": "complete",
        "output_root": str(output),
        "record_count": len(applied),
        "provenance": str(
            output / "queue_reactive_augmentation_provenance.json"
        ),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline-root", type=pathlib.Path, required=True)
    result.add_argument("--augmentation-root", type=pathlib.Path, required=True)
    result.add_argument("--output-root", type=pathlib.Path, required=True)
    result.add_argument(
        "--copy-files", action="store_true",
        help="physically copy the baseline instead of using same-filesystem hard links",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = apply(parser().parse_args(argv))
    except (ApplicationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"queue-reactive augmentation application failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
