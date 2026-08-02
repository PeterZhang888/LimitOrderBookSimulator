#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Relocate certified pooled evidence for queue-reactive model development.

The V19 evidence bundle records absolute Seagull paths.  This helper creates a
new, explicitly non-certified development input tree without mutating any
hash-bound evidence.  It also emits the training-only queue-depth targets and
the frozen stratified symbol list needed by the extended ITCH extractor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from collections.abc import Iterable, Mapping, Sequence


class PreparationError(RuntimeError):
    """Raised when source evidence is incomplete or internally inconsistent."""


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fields or not rows:
        raise PreparationError(f"empty CSV: {path}")
    return fields, rows


def write_csv(path: pathlib.Path, fields: Sequence[str],
              rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def require_columns(path: pathlib.Path, fields: Sequence[str],
                    required: set[str]) -> None:
    missing = sorted(required.difference(fields))
    if missing:
        raise PreparationError(f"{path} lacks columns: {', '.join(missing)}")


def canonical_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        raise PreparationError("empty symbol")
    return symbol


def relocate_rows(rows: list[dict[str, str]], pool_root: pathlib.Path,
                  cluster_by_symbol: Mapping[str, int]) -> list[dict[str, str]]:
    relocated: list[dict[str, str]] = []
    observed: set[str] = set()
    for expected_id, source_row in enumerate(rows):
        row = dict(source_row)
        symbol = canonical_symbol(row["symbol"])
        if symbol in observed:
            raise PreparationError(f"duplicate universe symbol: {symbol}")
        observed.add(symbol)
        if int(row["book_id"]) != expected_id:
            raise PreparationError("universe book_id values are not contiguous")
        if symbol not in cluster_by_symbol:
            raise PreparationError(f"cluster assignment missing {symbol}")

        data_basename = pathlib.Path(row["data_dir"]).name
        rates_basename = pathlib.Path(row["hawkes_rates_file"]).name
        data_dir = (pool_root / "pooled_data" / data_basename).resolve()
        rates_file = data_dir / rates_basename
        if not data_dir.is_dir():
            raise PreparationError(f"relocated data directory is absent: {data_dir}")
        if not rates_file.is_file():
            raise PreparationError(f"relocated rate file is absent: {rates_file}")
        row["symbol"] = symbol
        row["data_dir"] = str(data_dir)
        row["hawkes_rates_file"] = str(rates_file)
        row["cluster_id"] = str(cluster_by_symbol[symbol])
        relocated.append(row)
    if observed != set(cluster_by_symbol):
        extra = sorted(set(cluster_by_symbol).difference(observed))
        raise PreparationError(
            "cluster assignments contain symbols outside the universe: "
            + ", ".join(extra[:10])
        )
    return relocated


def reindex(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        row["book_id"] = str(index)
        result.append(row)
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    pool_root = pathlib.Path(args.pool_root).expanduser().resolve()
    assignments_path = pathlib.Path(args.cluster_assignments).expanduser().resolve()
    sample_path = pathlib.Path(args.validation_sample).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    training_path = pool_root / "pooled_training_universe.csv"
    heldout_path = pool_root / "heldout_common.csv"

    assignment_fields, assignment_rows = read_csv(assignments_path)
    require_columns(assignments_path, assignment_fields, {"symbol", "cluster_id"})
    cluster_by_symbol: dict[str, int] = {}
    for row in assignment_rows:
        symbol = canonical_symbol(row["symbol"])
        cluster = int(row["cluster_id"])
        if cluster < 0 or symbol in cluster_by_symbol:
            raise PreparationError(f"invalid cluster assignment for {symbol}")
        cluster_by_symbol[symbol] = cluster

    training_fields, training_rows = read_csv(training_path)
    heldout_fields, heldout_rows = read_csv(heldout_path)
    required = {
        "book_id", "symbol", "data_dir", "hawkes_rates_file",
        "target_mean_bid_depth", "target_mean_ask_depth",
    }
    require_columns(training_path, training_fields, required)
    require_columns(heldout_path, heldout_fields, required)
    if training_fields != heldout_fields:
        raise PreparationError("training and held-out universe schemas differ")

    training = relocate_rows(training_rows, pool_root, cluster_by_symbol)
    heldout = relocate_rows(heldout_rows, pool_root, cluster_by_symbol)
    if [row["symbol"] for row in training] != [row["symbol"] for row in heldout]:
        raise PreparationError("training and held-out symbol order differs")

    sample_fields, sample_rows = read_csv(sample_path)
    require_columns(sample_path, sample_fields, {"symbol", "cluster_id"})
    sample_symbols: list[str] = []
    for row in sample_rows:
        symbol = canonical_symbol(row["symbol"])
        if symbol in sample_symbols:
            raise PreparationError(f"duplicate stratified symbol: {symbol}")
        if symbol not in cluster_by_symbol:
            raise PreparationError(f"stratified symbol is outside universe: {symbol}")
        if int(row["cluster_id"]) != cluster_by_symbol[symbol]:
            raise PreparationError(f"stratified cluster mismatch for {symbol}")
        sample_symbols.append(symbol)
    sample_set = set(sample_symbols)
    training_sample = reindex(row for row in training if row["symbol"] in sample_set)
    heldout_sample = reindex(row for row in heldout if row["symbol"] in sample_set)
    if len(training_sample) != len(sample_symbols):
        raise PreparationError("not every stratified symbol was recovered")

    output_fields = list(training_fields)
    if "cluster_id" not in output_fields:
        output_fields.append("cluster_id")
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "training_universe": output_dir / "pooled_training_universe_local.csv",
        "heldout_universe": output_dir / "heldout_common_local.csv",
        "training_sample": output_dir / "pooled_training_sample_local.csv",
        "heldout_sample": output_dir / "heldout_sample_local.csv",
        "state_targets": output_dir / "training_state_targets.csv",
        "symbols": output_dir / "stratified_symbols.txt",
    }

    if args.value_policy:
        policy_path = pathlib.Path(args.value_policy).expanduser().resolve()
        policy_fields, policy_rows = read_csv(policy_path)
        require_columns(policy_path, policy_fields, {"symbol"})
        by_symbol = {
            canonical_symbol(row["symbol"]): dict(row) for row in policy_rows
        }
        if len(by_symbol) != len(policy_rows):
            raise PreparationError("value-agent policy contains duplicate symbols")
        missing = [symbol for symbol in sample_symbols if symbol not in by_symbol]
        if missing:
            raise PreparationError(
                "value-agent policy misses stratified symbols: "
                + ", ".join(missing[:10])
            )
        paths["sample_value_policy"] = output_dir / "sample_value_agent_policy.csv"
        write_csv(
            paths["sample_value_policy"], policy_fields,
            (by_symbol[symbol] for symbol in sample_symbols),
        )
    write_csv(paths["training_universe"], output_fields, training)
    write_csv(paths["heldout_universe"], output_fields, heldout)
    write_csv(paths["training_sample"], output_fields, training_sample)
    write_csv(paths["heldout_sample"], output_fields, heldout_sample)
    write_csv(
        paths["state_targets"],
        ["symbol", "cluster_id", "target_mean_bid_depth", "target_mean_ask_depth"],
        ({
            "symbol": row["symbol"],
            "cluster_id": row["cluster_id"],
            "target_mean_bid_depth": row["target_mean_bid_depth"],
            "target_mean_ask_depth": row["target_mean_ask_depth"],
        } for row in training_sample),
    )
    paths["symbols"].write_text(
        "".join(f"{symbol}\n" for symbol in sample_symbols), encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "role": "noncertified_queue_reactive_development_inputs",
        "source_pool_root": str(pool_root),
        "source_training_sha256": sha256_file(training_path),
        "source_heldout_sha256": sha256_file(heldout_path),
        "source_cluster_assignments_sha256": sha256_file(assignments_path),
        "source_validation_sample_sha256": sha256_file(sample_path),
        "universe_symbol_count": len(training),
        "stratified_symbol_count": len(training_sample),
        "cluster_count": len(set(cluster_by_symbol.values())),
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "warning": (
            "Relocation changes source bytes and is for model development only; "
            "it is not a replacement for the hash-bound V19 evidence."
        ),
    }
    manifest_path = output_dir / "development_input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--cluster-assignments", required=True)
    parser.add_argument("--validation-sample", required=True)
    parser.add_argument("--value-policy")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    try:
        result = run(build_parser().parse_args())
    except Exception as error:  # noqa: BLE001 - concise command-line boundary
        raise SystemExit(f"queue-reactive input preparation failed: {error}") from error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
