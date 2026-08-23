# Distributed Multi-Asset Limit Order Book Simulator

> **Original data source.** The empirical inputs were derived from the
> [official Nasdaq TotalView--ITCH archive](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/).
> Raw ITCH files are not included and remain subject to Nasdaq's terms.

This repository contains the simulator and experiment scripts used in the
thesis. Each asset has a complete price--time-priority limit order book.
Background order flow, liquidity agents, value agents and a shared market
maker interact across the simulated market. MPI distributes complete books
among ranks, with optional OpenMP parallelism within each rank.

## Build on Seagull

Run these commands separately on a Seagull login node:

```bash
git clone https://github.com/PeterZhang888/LimitOrderBookSimulator.git
cd LimitOrderBookSimulator
bash scripts/build_seagull.sh
```

The build runs the tests and creates:

```text
build-mpi/lob_mpi
build-openmp/lob_openmp
```

To submit selected full-session validation cases covering every experiment:

```bash
bash scripts/submit_seagull_validation.sh
bash scripts/check_seagull_validation.sh
```

Validation uses at most two nodes per job. Results are written below
`results/runs/release_validation_<timestamp>/`.

## Run an experiment

Every experiment directory contains:

- `compile.sh`, which builds the shared executables;
- `submit.sh`, which submits that experiment.

For example:

```bash
bash experiments/02_weak_scaling/compile.sh
bash experiments/02_weak_scaling/submit.sh
```

If `scripts/build_seagull.sh` has already completed, the experiment-specific
compile command is not needed. All experiments use the same executables.

| Directory | Experiment |
|---|---|
| `00_full_synthetic` | Complete 10,000-book synthetic session |
| `01_strong_scaling` | Synthetic strong scaling |
| `02_weak_scaling` | Synthetic weak scaling |
| `03_empirical_scaling` | Scaling of the 1,480-book empirical workload |
| `04_rank_ownership` | Cyclic and activity-weighted MPI ownership |
| `05_observation_buffering` | Synchronous and buffered observations |
| `06_mpi_openmp` | Permanent-ownership MPI--OpenMP decomposition |
| `07_risk_collectives` | Blocking, non-blocking and lookahead collectives |
| `08_stylised_facts` | Simulated return panels for stylised-fact analysis |
| `09_inventory_policy` | Shared Market Maker inventory policies |

Formal campaigns use the allocations and repetitions declared by their
submission scripts. See [the experiment guide](experiments/README.md) for the
individual configurations.

## Results

Simulation output is written under `results/runs/`, and scheduler logs are
written under `slurm/`. Each job records its execution environment. Every
repetition writes a console summary, boundary metrics and per-asset results.

Summarise a completed campaign with:

```bash
python3 scripts/summarize_results.py results/runs/CAMPAIGN_DIRECTORY
```

Compare two scientific CSV files with:

```bash
python3 scripts/compare_scientific_outputs.py reference.csv treatment.csv
```

## Source layout

| Path | Contents |
|---|---|
| `src/exchange/` | Limit order book and background-flow implementation |
| `src/simulation/` | Distributed event loop, agents, MPI and OpenMP execution |
| `src/main.cpp` | Command-line interface and program entry point |
| `data/empirical/` | Frozen inputs for the 1,480-book experiments |
| `examples/synthetic/` | Artificial input for the synthetic experiments |
| `experiments/` | Experiment build and submission scripts |
| `scripts/` | Build, validation and analysis utilities |

## Data and documentation

The repository includes the frozen inputs needed to run the empirical
experiments. It does not include raw Nasdaq messages or the derived empirical
one-second return panel used in the final comparison figure. Experiment 08
regenerates the simulated return panels.

- [Data inputs and provenance](DATA.md)
- [Experiment guide](experiments/README.md)
- [Reproducibility](REPRODUCIBILITY.md)
- [Testing](TESTING.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Use of the
software is governed by [`LICENSE`](LICENSE).
