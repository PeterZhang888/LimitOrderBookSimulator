# Whole-Book MPI Limit-Order-Book Simulator

C++20 simulator for causally coupled, multi-asset limit-order books. Each book
uses price--time priority and is owned by one MPI rank. Cross-asset state is
updated at deterministic decision boundaries through collective communication.

## Components

| Path | Purpose |
|---|---|
| `include/`, `src/` | Production simulator, agents, calibration support and MPI execution |
| `scripts/` | ITCH extraction, empirical pooling, clustering, calibration, validation and analysis |
| `config/` | Small model configurations and the fixed 1,480-symbol cohort |
| `tests/` | C++ correctness tests and Python workflow-contract tests |
| `submit_*.sh` | Slurm launchers for calibration, validation and case-study execution |
| `results/final-case-study/` | Compact timing and treatment summaries from the final campaign |
| `docs/case-study/` | LaTeX tables, figures and the case-study analysis report |

The production build uses `include/` and `src/`. Historical sources retained
under `Draft/` and `include/*_hpp/` are excluded from CMake targets.

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

## Executables

| Target | Function |
|---|---|
| `fragmented_mpi_lob` | Whole-book MPI simulator used by the final experiments |
| `sequential_multi_asset_lob` | One-process semantic reference |
| `exact_mpi_multi_asset_lob` | Exact distributed multi-asset execution |
| `batched_mpi_multi_asset_lob` | Batched-communication execution |
| `smc_abc_calibrate` | SMC-ABC calibration driver |
| `eligibility_evaluate` | Structural-validity evaluator |

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
5. Run rank-equivalence, scaling or financial treatments with
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

## Results

The final case-study summaries and rank-equivalence records are under
`results/final-case-study/`. The associated interpretation is in
`docs/case-study/case_study_analysis_report.md`.

No open-source licence has been selected; see [`LICENSE`](LICENSE).
