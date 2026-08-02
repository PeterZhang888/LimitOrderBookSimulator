#!/usr/bin/env python3
"""Rebase a hash-bound held-out target config onto its cluster data root."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
from typing import Mapping


DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def sha256(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def one_artifact(directory: pathlib.Path, pattern: str, label: str) -> pathlib.Path:
    matches = sorted(directory.glob(pattern))
    if pattern.startswith("market_targets_"):
        matches = [path for path in matches if "_window_" not in path.name]
    if len(matches) != 1:
        raise ValueError(
            f"{label} requires exactly one {pattern} in {directory}; "
            f"observed {len(matches)}"
        )
    return matches[0]


def prepare(
    *, provenance_path: pathlib.Path, output_path: pathlib.Path,
    expected_date: str,
) -> dict[str, object]:
    provenance_path = provenance_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not DATE_PATTERN.fullmatch(expected_date):
        raise ValueError("expected date must be YYYY-MM-DD")
    if not provenance_path.is_file():
        raise ValueError(f"pooling provenance is missing: {provenance_path}")
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    if payload.get("heldout_date") != expected_date:
        raise ValueError(
            f"pooling provenance heldout_date is {payload.get('heldout_date')!r}, "
            f"expected {expected_date}"
        )
    heldout = payload.get("heldout")
    if not isinstance(heldout, Mapping):
        raise ValueError("pooling provenance lacks heldout metadata")
    if heldout.get("heldout_role") != "opening_state_and_validation_targets_only":
        raise ValueError("pooling provenance has an incompatible heldout role")
    if heldout.get("background_inputs_inherited_from_pooled") is not True:
        raise ValueError("heldout runtime background is not frozen from training")

    source_config = pathlib.Path(
        str(heldout.get("source_config", ""))
    ).expanduser().resolve()
    if not source_config.is_file() or source_config.stat().st_size == 0:
        raise ValueError(f"heldout source config is missing or empty: {source_config}")
    expected_source_hash = str(heldout.get("source_config_sha256", ""))
    observed_source_hash = sha256(source_config)
    if observed_source_hash != expected_source_hash:
        raise ValueError(
            "heldout source config SHA-256 mismatch: "
            f"expected {expected_source_hash}, observed {observed_source_hash}"
        )
    target_root = pathlib.Path(
        str(heldout.get("target_root", ""))
    ).expanduser().resolve()
    if not target_root.is_dir():
        raise ValueError(f"heldout target root is not a directory: {target_root}")

    with source_config.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if "symbol" not in fields or "data_dir" not in fields or not rows:
        raise ValueError("heldout source config requires nonempty symbol and data_dir rows")
    if "target_data_dir" not in fields:
        fields.append("target_data_dir")
    expected_count = int(payload.get("common_symbol_count", 0))
    if expected_count <= 0 or len(rows) != expected_count:
        raise ValueError(
            f"heldout source config has {len(rows)} rows; provenance expects "
            f"{expected_count}"
        )

    seen: set[str] = set()
    artifact_records: list[dict[str, str]] = []
    for line, row in enumerate(rows, start=2):
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol or symbol in seen:
            raise ValueError(f"invalid or duplicate symbol at {source_config}:{line}")
        seen.add(symbol)
        original = str(row.get("target_data_dir", "")).strip() \
            or str(row.get("data_dir", "")).strip()
        basename = pathlib.Path(original).name
        if not basename or basename in {".", ".."}:
            raise ValueError(f"invalid empirical directory at {source_config}:{line}")
        target_dir = (target_root / basename).resolve()
        if target_dir.parent != target_root or not target_dir.is_dir():
            raise ValueError(
                f"rebased target directory is missing for {symbol}: {target_dir}"
            )
        target_file = one_artifact(
            target_dir, "market_targets_*.csv", f"target file for {symbol}"
        )
        manifest_file = one_artifact(
            target_dir, "itch_manifest_*.json", f"manifest for {symbol}"
        )
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if str(manifest.get("symbol", "")).strip().upper() != symbol:
            raise ValueError(f"manifest symbol mismatch for {symbol}: {manifest_file}")
        if manifest.get("trading_date") != expected_date:
            raise ValueError(
                f"manifest date mismatch for {symbol}: "
                f"{manifest.get('trading_date')!r} != {expected_date!r}"
            )
        row["target_data_dir"] = str(target_dir)
        artifact_records.append({
            "symbol": symbol,
            "target_directory": str(target_dir),
            "target_sha256": sha256(target_file),
            "manifest_sha256": sha256(manifest_file),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_path)
    record = {
        "schema_version": 1,
        "status": "heldout_target_paths_rebased_and_verified",
        "expected_date": expected_date,
        "symbol_count": len(rows),
        "pooling_provenance": {
            "path": str(provenance_path), "sha256": sha256(provenance_path),
        },
        "hash_bound_source_config": {
            "path": str(source_config), "sha256": observed_source_hash,
            "modified": False,
        },
        "target_root": str(target_root),
        "output_config": {"path": str(output_path), "sha256": sha256(output_path)},
        "artifacts": artifact_records,
    }
    record_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
    temporary_record = record_path.with_name(record_path.name + f".tmp.{os.getpid()}")
    temporary_record.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_record, record_path)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooling-provenance", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--expected-date", required=True)
    args = parser.parse_args()
    try:
        result = prepare(
            provenance_path=args.pooling_provenance,
            output_path=args.output,
            expected_date=args.expected_date,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"heldout target preparation failed: {error}") from error
    print(json.dumps({
        "status": result["status"],
        "symbol_count": result["symbol_count"],
        "output_config": result["output_config"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
