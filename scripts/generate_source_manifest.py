#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for a source-only release tree."""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib


EXCLUDED_DIRECTORY_NAMES = {
    ".git", ".pytest_cache", "__pycache__", "results", "slurm",
}
EXCLUDED_DIRECTORY_PREFIXES = ("build",)


def included(path: pathlib.Path, root: pathlib.Path) -> bool:
    relative = path.relative_to(root)
    if path.name == "SOURCE_MANIFEST.sha256" or path.name.startswith("._"):
        return False
    return not any(
        part in EXCLUDED_DIRECTORY_NAMES
        or part.startswith(EXCLUDED_DIRECTORY_PREFIXES)
        for part in relative.parts[:-1]
    )


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(root: pathlib.Path, output: pathlib.Path) -> int:
    root = root.expanduser().resolve()
    output = output.expanduser().resolve()
    if not root.is_dir() or output.parent != root:
        raise SystemExit("root must exist and output must be directly inside it")
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and included(path, root)
    )
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for path in files:
            relative = path.relative_to(root).as_posix()
            handle.write(f"{digest(path)}  ./{relative}\n")
    os.replace(temporary, output)
    print(f"manifest={output}")
    print(f"files={len(files)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    return run(args.root, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
