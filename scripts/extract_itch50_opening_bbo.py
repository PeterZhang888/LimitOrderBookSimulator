#!/usr/bin/env python3
"""Extract one preconfigured BBO/depth snapshot for several ITCH symbols."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import pathlib
from typing import BinaryIO

from extract_itch50_symbol import (
    VisibleBook,
    parse_clock_ns,
    read_exact,
    stock_text,
    timestamp_ns,
)


def extract(input_path: pathlib.Path, symbols: list[str], clock: str) -> list[dict[str, object]]:
    requested = {symbol.upper() for symbol in symbols}
    locate_to_symbol: dict[int, str] = {}
    books = {symbol: VisibleBook() for symbol in requested}
    target_ns = parse_clock_ns(clock)

    raw = input_path.open("rb")
    try:
        magic = raw.read(2)
        raw.seek(0)
        decoded: BinaryIO = gzip.GzipFile(fileobj=raw, mode="rb") if magic == b"\x1f\x8b" else raw
        stream = io.BufferedReader(decoded, buffer_size=8 * 1024 * 1024)
        while True:
            length_bytes = stream.read(2)
            if not length_bytes:
                break
            if len(length_bytes) != 2:
                raise EOFError("truncated two-byte ITCH record length")
            length = int.from_bytes(length_bytes, "big", signed=False)
            if length == 0:
                break
            message = read_exact(stream, length)
            kind = chr(message[0])

            if kind == "R" and len(message) == 39:
                symbol = stock_text(message[11:19])
                if symbol in requested:
                    locate_to_symbol[int.from_bytes(message[1:3], "big", signed=False)] = symbol
                continue

            if len(message) < 11:
                continue
            event_time_ns = timestamp_ns(message)
            if event_time_ns >= target_ns and len(locate_to_symbol) == len(requested):
                break

            symbol = locate_to_symbol.get(int.from_bytes(message[1:3], "big", signed=False))
            if symbol is None:
                continue
            book = books[symbol]

            if (kind == "A" and len(message) == 36) or (kind == "F" and len(message) == 40):
                reference = int.from_bytes(message[11:19], "big", signed=False)
                side = chr(message[19])
                shares = int.from_bytes(message[20:24], "big", signed=False)
                price = int.from_bytes(message[32:36], "big", signed=False)
                if stock_text(message[24:32]) == symbol:
                    book.add(reference, side, shares, price)
            elif (kind == "E" and len(message) == 31) or (kind == "C" and len(message) == 36):
                reference = int.from_bytes(message[11:19], "big", signed=False)
                executed = int.from_bytes(message[19:23], "big", signed=False)
                book.reduce(reference, executed)
            elif kind == "X" and len(message) == 23:
                reference = int.from_bytes(message[11:19], "big", signed=False)
                cancelled = int.from_bytes(message[19:23], "big", signed=False)
                book.reduce(reference, cancelled)
            elif kind == "D" and len(message) == 19:
                book.delete(int.from_bytes(message[11:19], "big", signed=False))
            elif kind == "U" and len(message) == 35:
                old_reference = int.from_bytes(message[11:19], "big", signed=False)
                new_reference = int.from_bytes(message[19:27], "big", signed=False)
                shares = int.from_bytes(message[27:31], "big", signed=False)
                price = int.from_bytes(message[31:35], "big", signed=False)
                book.replace(old_reference, new_reference, shares, price)
    finally:
        raw.close()

    missing = requested.difference(locate_to_symbol.values())
    if missing:
        raise RuntimeError("symbols absent from Stock Directory: " + ", ".join(sorted(missing)))

    rows: list[dict[str, object]] = []
    for symbol in symbols:
        book = books[symbol.upper()]
        bid = book.best_bid()
        ask = book.best_ask()
        if bid <= 0 or ask <= bid:
            raise RuntimeError(f"{symbol} is not two-sided at {clock}")
        rows.append({
            "symbol": symbol.upper(),
            "clock": clock,
            "best_bid_ticks": bid,
            "best_ask_ticks": ask,
            "best_bid_depth": book.best_depth("B"),
            "best_ask_depth": book.best_depth("S"),
            "mid_price_ticks": 0.5 * (bid + ask),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--clock", default="09:30:00")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = extract(pathlib.Path(args.input).resolve(), args.symbols, args.clock)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
