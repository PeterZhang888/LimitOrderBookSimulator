# Testing

`scripts/build_seagull.sh` builds the MPI and MPI-free OpenMP executables and
runs the compiled correctness tests. The tests cover order matching, reports,
background flow, cancellation fallback, queue-policy loading, model semantics,
Shared Market Maker coverage and OpenMP equivalence when OpenMP is available.

For experiment equivalence, compare the complete scientific CSV contents,
processed-order count, trade count, boundary count, Shared Market Maker cash,
inventory and terminal accounting values. Do not infer equivalence from wall
time or from one summary value.

`scripts/summarize_results.py` reads the complete simulator result line from
each repetition and reports all raw rows before calculating timing summaries.
