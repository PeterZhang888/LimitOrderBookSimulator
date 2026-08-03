# Whole-Book MPI Limit-Order-Book Simulator

C++20 simulator for causally coupled, multi-asset limit-order books. Each book
uses price--time priority and is owned by one MPI rank. Cross-asset state is
updated at deterministic decision boundaries through collective communication.

## Components

| Path | Purpose |
|---|---|
| `include/`, `src/` | Production whole-book MPI simulator |
| `scripts/` | ITCH extraction, pooling, calibration, validation and final analysis |
| `config/` | Test configuration and the fixed 1,480-symbol cohort |
| `tests/` | C++ correctness tests and Python workflow-contract tests |
| `submit_*.sh` | Slurm launchers for calibration, validation and case-study execution |
| `results/final-case-study/` | Compact, hash-traceable summaries from the completed inventory-stress campaign |
| `docs/case-study/` | Experiment specification, result audit, LaTeX text and figures |

The source tree contains only the executable dependency closure used by the
calibration, validation, scaling and inventory-stress experiments.

## Build

Requirements:

- CMake 3.20 or later;
- a C++20 compiler;
- an MPI implementation for distributed execution;
- Python 3.10 or later for workflow utilities.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOB_REQUIRE_MPI=ON \
  -DLOB_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build -LE 'empirical|mpi' --output-on-failure
```

Set `LOB_REQUIRE_MPI=OFF` only for single-process development and unit testing.

The build produces one executable, `fragmented_mpi_lob`. A one-rank run of
that same executable is the semantic reference for MPI rank-equivalence tests.

## Empirical workflow

Raw Nasdaq TotalView--ITCH archives are not distributed with the repository.
The main workflow is:

1. Extract book events and empirical targets with
   `scripts/extract_itch50_symbols.py`.
2. Pool the five training sessions with
   `submit_five_day_pooled_training.sh`.
3. Form liquidity clusters and select behavioural parameters with
   `submit_cluster_value_agent_calibration.sh`.
4. Validate the frozen model with
   `submit_queue_reactive_full_validation_hpc.sh`.
5. Run rank-equivalence, mechanism, performance and financial phases of the
   same liquidity-shock experiment with
   `submit_queue_reactive_case_study.sh`.

Cluster jobs must be submitted with `sbatch`; the login node is used only for
data inspection, packaging and job submission. Dataset requirements and path
conventions are specified in [`DATA.md`](DATA.md).

## Determinism

Logical identifiers and random streams are independent of MPI ownership.
Fixed-point collective reductions and canonical state hashing allow a run at
one rank to be compared with the same realization at multiple ranks. The case
launcher performs this rank-equivalence preflight before production treatments.

## Verification

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
sha256sum -c SOURCE_MANIFEST.sha256
```

Some integration tests require the external empirical directories. See
[`TESTING.md`](TESTING.md) for the verified source-only subset and
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full cluster workflow.

## Final experiment evidence

Production outputs are written below `results/seagull/` and are not committed
in full. The completed 1,480-book inventory-stress campaign contains 200
full-session financial paths and six implementation/preflight paths.
The compact result tables, mechanism certificate, archive digest and execution
audit are retained in `results/final-case-study/`; the full hash-bound archive
is external because it expands to more than 8 GB. The principal result is a
small withdrawal of unshocked top-of-book depth during the first seconds after
the intervention, followed by recovery within tens of seconds. No persistent
30-minute depth or spread effect is established.

The repository is distributed under the terms in [`LICENSE`](LICENSE).
