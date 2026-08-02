# Whole-Book MPI Decomposition for Coupled Limit-Order-Book Simulation

Research code accompanying Peter Zhang's HPC master's thesis. The repository
contains the C++20 limit-order-book simulator, MPI whole-book decomposition,
ITCH preprocessing and calibration utilities, validation tools, Slurm launchers,
tests, and the audited R36 case-study analysis.

## Scientific scope

The final model assigns each complete price--time-priority book to one MPI rank.
Book matching remains sequential within a book, while ranks exchange the shared
market maker's aggregate exposure at fixed decision boundaries. The empirical
background is informed by Nasdaq TotalView--ITCH observations and a queue-reactive
Hawkes specification. Behavioural policies are selected by liquidity cluster.

The successful R36 campaign used 1,480 books, 32 ranks, a 23,400-second session,
a one-second decision window and 40 paired shock/control paths. Its audited
scientific conclusion is deliberately limited: the direct shocks were executed,
but the global shared market maker had stopped quoting before the shock, so the
run does not establish cross-asset contagion. See
[`docs/case-study/case_study_analysis_report.md`](docs/case-study/case_study_analysis_report.md).

## Repository map

- `include/`, `src/`: simulator and MPI implementation.
- `Draft/`, `include/*_hpp/` and the retained legacy placeholders: early
  development sources preserved from the original GitHub history. They are not
  deleted or rewritten into the production layout.
- `scripts/`: ITCH extraction, pooling, clustering, calibration, validation,
  experiment orchestration and analysis.
- `tests/`: C++ and Python correctness tests.
- `config/`: small configuration inputs and the frozen 1,480-symbol identity list.
- `submit_*.sh`: Slurm workflows used during model development.
- `submit_queue_reactive_case_study.sh`: final R36 Seagull case-study launcher.
- `docs/case-study/`: audited chapter text, figures and analysis outputs.
- `TESTING.md`: release-audit results and data/source-dependent test limits.

## Build

Requirements are CMake 3.20+, a C++20 compiler and, for distributed execution,
an MPI implementation. Python utilities use the Python standard library.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLOB_REQUIRE_MPI=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Set `LOB_REQUIRE_MPI=OFF` for single-process correctness mode. Some integration
tests require derived ITCH inputs that are intentionally not committed.
The precise verification status of this public-source collection is recorded in
[`TESTING.md`](TESTING.md); it distinguishes code failures from unavailable
empirical fixtures and from one known historical source-export gap.

## Empirical data

Raw Nasdaq ITCH archives and the large derived symbol directories are excluded
because of their size and redistribution conditions. The repository retains
the extraction and calibration code, cohort identity, small configurations and
hash-bound result summaries. See [`DATA.md`](DATA.md).

## Reproducing the R36 case

The case launcher expects the pooled empirical root, selected-policy root and
derived data root to exist on the execution system. The successful Seagull run
used external hash-bound artifacts and cannot be reproduced from this source
repository alone. Instructions and the precise source-export limitation are in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Provenance and attribution

This project was developed for Peter Zhang's thesis with OpenAI ChatGPT/Codex
assistance. No vendored or copied third-party source code was identified in the
available source history. Published models, file-format specifications,
dependencies and external data are acknowledged without implying that their
authors wrote this implementation. See [`PROVENANCE.md`](PROVENANCE.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

No open-source licence has been selected. The default copyright position is
therefore all rights reserved; see [`LICENSE`](LICENSE).

## Development history

This tree extends the existing `PeterZhang888/LimitOrderBookSimulator`
repository. Commit `0c14558` is the final pre-integration website-upload state;
subsequent commits separately introduce the production core, empirical
workflows, tests, R36 evidence and release documentation. Consequently, Git can
show both the early implementation and every thesis-integration change without
requiring a force push or replacement repository.
