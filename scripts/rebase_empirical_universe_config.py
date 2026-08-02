#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Create a verified, location-independent copy of an empirical config CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re


DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one_manifest(directory: pathlib.Path) -> pathlib.Path:
    matches = sorted(directory.glob("itch_manifest_*.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one ITCH manifest in {directory}; "
            f"observed {len(matches)}"
        )
    return matches[0]


def rebase(
    *, input_config: pathlib.Path, empirical_root: pathlib.Path,
    output_config: pathlib.Path, expected_date: str,
) -> dict[str, object]:
    if not DATE_PATTERN.fullmatch(expected_date):
        raise ValueError("expected date must be YYYY-MM-DD")
    input_config = input_config.expanduser().resolve()
    empirical_root = empirical_root.expanduser().resolve()
    output_config = output_config.expanduser().resolve()
    if not input_config.is_file() or input_config.stat().st_size == 0:
        raise ValueError(f"input config is missing or empty: {input_config}")
    if not empirical_root.is_dir():
        raise ValueError(f"empirical root is not a directory: {empirical_root}")
    if input_config == output_config:
        raise ValueError("input and output configs must be different files")

    with input_config.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    required = {"book_id", "symbol", "data_dir", "hawkes_rates_file"}
    missing = sorted(required.difference(fields))
    if missing or not rows:
        raise ValueError(
            f"config is empty or lacks required fields: {', '.join(missing)}"
        )

    compact_date = expected_date.replace("-", "")
    seen_symbols: set[str] = set()
    records: list[dict[str, str]] = []
    for line_number, row in enumerate(rows, start=2):
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol or symbol in seen_symbols:
            raise ValueError(
                f"invalid or duplicate symbol at {input_config}:{line_number}"
            )
        seen_symbols.add(symbol)

        old_data_dir = pathlib.Path(str(row.get("data_dir", "")).strip())
        directory_name = old_data_dir.name
        expected_suffix = f"_{symbol.lower()}"
        if (
            not directory_name
            or compact_date not in directory_name
            or not directory_name.lower().endswith(expected_suffix)
        ):
            raise ValueError(
                f"cannot identify the empirical directory for {symbol} from "
                f"{old_data_dir}"
            )
        data_dir = (empirical_root / directory_name).resolve()
        if data_dir.parent != empirical_root or not data_dir.is_dir():
            raise ValueError(f"rebased empirical directory is missing: {data_dir}")

        old_rates = pathlib.Path(str(row.get("hawkes_rates_file", "")).strip())
        rates_name = old_rates.name
        if not rates_name:
            raise ValueError(f"missing Hawkes-rate basename for {symbol}")
        rates_file = (data_dir / rates_name).resolve()
        if rates_file.parent != data_dir or not rates_file.is_file():
            raise ValueError(f"rebased Hawkes-rate file is missing: {rates_file}")

        manifest_file = one_manifest(data_dir)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if str(manifest.get("symbol", "")).strip().upper() != symbol:
            raise ValueError(f"manifest symbol mismatch for {symbol}: {manifest_file}")
        if manifest.get("trading_date") != expected_date:
            raise ValueError(
                f"manifest date mismatch for {symbol}: "
                f"{manifest.get('trading_date')!r} != {expected_date!r}"
            )

        row["data_dir"] = str(data_dir)
        row["hawkes_rates_file"] = str(rates_file)
        if "target_data_dir" in fields:
            row["target_data_dir"] = str(data_dir)
        records.append({
            "symbol": symbol,
            "data_dir": str(data_dir),
            "manifest_sha256": sha256(manifest_file),
            "hawkes_rates_sha256": sha256(rates_file),
        })

    output_config.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_config.with_name(
        output_config.name + f".tmp.{os.getpid()}"
    )
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_config)

    result = {
        "schema_version": 1,
        "status": "empirical_config_rebased_and_verified",
        "expected_date": expected_date,
        "symbol_count": len(rows),
        "source_config": {
            "path": str(input_config),
            "sha256": sha256(input_config),
            "modified": False,
        },
        "empirical_root": str(empirical_root),
        "output_config": {
            "path": str(output_config),
            "sha256": sha256(output_config),
        },
        "records": records,
    }
    provenance = output_config.with_suffix(
        output_config.suffix + ".provenance.json"
    )
    temporary_provenance = provenance.with_name(
        provenance.name + f".tmp.{os.getpid()}"
    )
    temporary_provenance.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_provenance, provenance)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-config", required=True, type=pathlib.Path)
    parser.add_argument("--empirical-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--expected-date", required=True)
    args = parser.parse_args()
    try:
        result = rebase(
            input_config=args.input_config,
            empirical_root=args.empirical_root,
            output_config=args.output,
            expected_date=args.expected_date,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"empirical-config rebasing failed: {error}") from error
    print(json.dumps({
        "status": result["status"],
        "expected_date": result["expected_date"],
        "symbol_count": result["symbol_count"],
        "output_config": result["output_config"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
