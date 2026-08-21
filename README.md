# Distributed Multi-Asset Limit Order Book Simulator

This repository contains the final simulator used by the thesis. Each asset
has one complete order-level limit order book. MPI assigns complete books to
ranks; OpenMP can process different books owned by the same rank. A book is
never divided between ranks or threads.

## Source layout

| File | Responsibility |
|---|---|
| `src/exchange/LimitOrderBook.cpp` | One complete book: queues, matching, cancellations and trades |
| `src/exchange/BackgroundHawkesAgent.cpp` | Background order-message generation |
| `src/simulation/DistributedMarketSimulator.cpp` | Multi-asset event loop, agent decisions, book ownership, MPI boundaries and OpenMP worksharing |
| `src/main.cpp` | Command-line parsing, MPI initialisation, launch and final output |
| `include/mpi/MpiCompat.hpp` | One-process compatibility layer for the MPI-free OpenMP build |

## Compile on Seagull

From the repository root:

```bash
mkdir -p slurm results/seagull
bash scripts/build_seagull.sh
```

This produces:

```text
build-mpi/lob_mpi
build-openmp/lob_openmp
```

`lob_mpi` supports pure MPI and hybrid MPI--OpenMP. `lob_openmp` is the
one-process MPI-free OpenMP executable.

## Run a complete synthetic session

The artificial input under `examples/synthetic/` is included so the code can
run without Nasdaq data. For a full 10,000-book session:

```bash
mpirun -np 64 build-mpi/lob_mpi \
  --duration-seconds 23400 \
  --assets 10000 \
  --base-config examples/synthetic/templates.csv \
  --background-model legacy \
  --partition cyclic \
  --synchronous-observations \
  --disable-persistent-risk-collective \
  --shared-inventory-policy gross_pooled \
  --threads 1 \
  --metrics-csv results/synthetic_metrics.csv \
  --asset-summary-csv results/synthetic_assets.csv
```

Do not run a full session on a login node. Submit it through Slurm or use an
interactive compute allocation.

## Thesis experiments

The `experiments/` directory contains one full submission file per experiment.
All use the same executables; only the declared treatment flags differ.

```text
01_strong_scaling
02_weak_scaling
03_empirical_scaling
04_rank_ownership
05_observation_buffering
06_fused_metric_scans
07_mpi_openmp
08_risk_collectives
09_stylised_facts
10_inventory_policy
```

For empirical experiments, export the paths described in
`experiments/README.md`, then submit from the repository root. For example:

```bash
export PROJECT_DIR="$PWD"
export UNIVERSE_CONFIG=/path/to/frozen_universe.csv
export BACKGROUND_POLICY_CSV=/path/to/background_policy.csv
export VALUE_POLICY_CSV=/path/to/value_agent_policy.csv
sbatch experiments/08_risk_collectives/submit_seagull.sh
```

Every performance treatment writes separate boundary-metric, per-asset and
console-output files. Scientific CSVs can be compared directly with:

```bash
python3 scripts/compare_scientific_outputs.py reference.csv treatment.csv
```

## Default comparison rule

Unless an experiment documents a necessary prerequisite, the unoptimised
control uses cyclic ownership (`asset_id mod MPI ranks`), synchronous
observations, blocking `MPI_Allreduce`, one thread per rank, and no scan,
lookahead or persistent-team optimisation. The lookahead experiment is the
one exception: all four cells use buffered observations because the exact
lookahead implementation requires them.

## Data

The repository does not redistribute Nasdaq ITCH data. See `DATA.md` for the
required empirical inputs and the included artificial alternative.
