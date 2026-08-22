#!/usr/bin/env python3
import collections
import math
import pathlib
import sys


def expand_cpu_list(value):
    cpus = set()
    for part in value.split(","):
        if "-" in part:
            first_text, last_text = part.split("-", 1)
            first = int(first_text)
            last = int(last_text)
            if last < first:
                raise ValueError("descending CPU range: {}".format(part))
            cpus.update(range(first, last + 1))
        else:
            cpus.add(int(part))
    return cpus


def main():
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: validate_cpu_placement.py FILE RANKS THREADS TASKS_PER_NODE"
        )
    path = pathlib.Path(sys.argv[1])
    ranks = int(sys.argv[2])
    threads = int(sys.argv[3])
    tasks_per_node = int(sys.argv[4])

    rows = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw.split("|", 2)
        if len(parts) != 3:
            raise SystemExit(
                "{}:{}: expected host|rank|cpu-list".format(path, line_number)
            )
        host, rank_text, cpu_text = parts
        try:
            rank = int(rank_text)
            cpus = expand_cpu_list(cpu_text)
        except ValueError as error:
            raise SystemExit("{}:{}: {}".format(path, line_number, error))
        if len(cpus) != threads:
            raise SystemExit(
                "rank {} on {} has {} CPUs, expected {}".format(
                    rank, host, len(cpus), threads
                )
            )
        rows.append((host, rank, cpus))

    if len(rows) != ranks:
        raise SystemExit("observed {} ranks, expected {}".format(len(rows), ranks))
    observed_ranks = sorted(rank for _, rank, _ in rows)
    if observed_ranks != list(range(ranks)):
        raise SystemExit("rank identifiers are incomplete or duplicated")

    by_host = collections.defaultdict(list)
    for host, rank, cpus in rows:
        by_host[host].append((rank, cpus))
    expected_hosts = int(math.ceil(float(ranks) / float(tasks_per_node)))
    if len(by_host) != expected_hosts:
        raise SystemExit(
            "observed {} nodes, expected {}".format(len(by_host), expected_hosts)
        )

    for host, assignments in by_host.items():
        if len(assignments) > tasks_per_node:
            raise SystemExit(
                "{} has {} ranks, maximum is {}".format(
                    host, len(assignments), tasks_per_node
                )
            )
        occupied = set()
        for rank, cpus in assignments:
            overlap = occupied.intersection(cpus)
            if overlap:
                raise SystemExit(
                    "rank {} on {} overlaps CPUs {}".format(
                        rank, host, sorted(overlap)
                    )
                )
            occupied.update(cpus)

    print(
        "CPU placement: PASS ({} ranks, {} threads/rank, {} nodes)".format(
            ranks, threads, len(by_host)
        )
    )


if __name__ == "__main__":
    main()
