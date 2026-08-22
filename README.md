# Distributed Multi-Asset Limit Order Book Simulator

> **Original data source.** The raw order-message files used to prepare the
> frozen empirical inputs were obtained from the
> [official Nasdaq TotalView--ITCH archive](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/).
> Raw ITCH files are not redistributed in this repository; access and use of
> Nasdaq data remain subject to Nasdaq's applicable terms.

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

On a Seagull login node:

```bash
git clone https://github.com/PeterZhang888/LimitOrderBookSimulator.git
cd LimitOrderBookSimulator

mkdir -p slurm results/seagull
bash scripts/build_seagull.sh
```

The build command first checks that all frozen runtime inputs are present. It
then requires OpenMP support, builds both executables and runs the compiled
correctness tests against both builds. No raw Nasdaq message files or
calibration run are required.

The Seagull scripts load
`openmpi/5.0.9-gcc-15.2.0-2irqibq`, the module used for the final release
tests. Set `OPENMPI_MODULE` only if the cluster administrators replace that
module and record the replacement with the results.

This produces:

```text
build-mpi/lob_mpi
build-openmp/lob_openmp
```

`lob_mpi` supports pure MPI and hybrid MPI--OpenMP. `lob_openmp` is the
one-process MPI-free OpenMP executable.

## Submit a complete synthetic session

The artificial input under `examples/synthetic/` is included so the code can
run without Nasdaq data. Submit the full 10,000-book, 23,400-second session
from the repository root:

```bash
sbatch experiments/00_full_synthetic/submit_seagull.sh
```

The job uses 16 nodes, 256 MPI ranks, one thread per rank and one complete
session. Results are written below `results/seagull/<job-id>/full_10000/`.
Do not run a full session on a login node.

## Thesis experiments

The `experiments/` directory contains one full submission file per experiment.
All use the same executables; only the declared treatment flags differ.

```text
00_full_synthetic
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

The strong-scaling file is a submission driver because each rank count needs a
different number of nodes. From the repository root, submit its complete
1--256-rank campaign with:

```bash
bash experiments/01_strong_scaling/submit_seagull.sh
```

All other experiment submission files are passed directly to `sbatch`. The
frozen empirical universe and policies are included under `data/empirical/`,
so no cluster-specific input paths need to be exported. A complete submission
sequence is:

```bash
mkdir -p slurm results/seagull
bash experiments/01_strong_scaling/submit_seagull.sh
sbatch experiments/02_weak_scaling/submit_seagull.sh
sbatch experiments/03_empirical_scaling/submit_seagull.sh
sbatch experiments/04_rank_ownership/submit_seagull.sh
sbatch experiments/05_observation_buffering/submit_seagull.sh
sbatch experiments/06_fused_metric_scans/submit_seagull.sh
sbatch experiments/07_mpi_openmp/submit_seagull.sh
sbatch experiments/08_risk_collectives/submit_seagull.sh
sbatch experiments/09_stylised_facts/submit_seagull.sh
sbatch experiments/10_inventory_policy/submit_seagull.sh
```

These are formal full-session jobs. The performance experiments use seven
repetitions by default. Experiment 07 creates its measured per-book scheduling
costs from one full preparation run before starting the timed OpenMP
comparisons; that preparation run is stored separately and is not included in
the reported comparisons.

The focused three-way OpenMP implementation comparison uses one executable,
one exclusive four-node allocation and seven counterbalanced blocks. Every run
uses 32 MPI ranks, two threads per rank, cyclic book ownership, synchronous
observations and the same empirical inputs. Only the OpenMP execution method
changes between all-phase worksharing, event-window-only worksharing and a
persistent task-based team:

```bash
sbatch experiments/07_mpi_openmp/submit_three_way_comparison.sh
```

The job directly compares all boundary-metric and per-asset output files and
stops if any scientific result differs.

Two smaller matched campaigns provide the pure-MPI reference at 64 physical
cores and the pure-MPI versus MPI-free OpenMP comparison at 16 physical
cores:

```bash
sbatch experiments/07_mpi_openmp/submit_64core_mpi_hybrid_pair.sh
sbatch experiments/07_mpi_openmp/submit_16core_mpi_openmp_pair.sh
```

The first pairs 64 MPI ranks with 32 ranks times two threads on the same four
nodes. The second pairs 16 MPI ranks with one process times 16 threads on the
same node. Both use seven alternating-order blocks, check that the two layouts
receive the same physical cores, and require exact agreement in per-asset
outputs, event counts and final accounting. Because MPI rank layouts combine
floating-point spread sums in different orders, the three derived spread
columns are compared with the narrow rule
\( |a-b|\leq5\times10^{-9}+10^{-12}\max(|a|,|b|) \); every other metric
column remains exact.

The shared Seagull runner also freezes the empirical-market controls used in
the thesis: a 23,400-second activity-normalisation horizon, empirical relative
quote and capacity sizing, three Shared Market Maker price levels, local
inventory scale 800, portfolio capacity 50 per asset, activation threshold
0.5 and minimum quote scale 0.05. The ordinary empirical experiments use the
selected quote multiplier 2.00; experiment 10 deliberately replaces that one
value across its declared participation sweep.

Every performance treatment writes separate boundary-metric, per-asset and
console-output files. Each job also writes `environment.txt`, containing its
compiler, MPI version, CPU description, allocation and loaded modules.
Scientific CSVs can be compared directly with:

```bash
python3 scripts/compare_scientific_outputs.py reference.csv treatment.csv
```

After a campaign finishes, collect every completed simulator row and generate
the raw and median/minimum/maximum timing tables with:

```bash
python3 scripts/summarize_results.py results/seagull/<job-or-campaign-directory>
```

The command writes `raw_results.csv` and `performance_summary.csv` below the
selected result directory. It does not discard slow repetitions.

## Default comparison rule

Unless an experiment documents a necessary prerequisite, the unoptimised
control uses cyclic ownership (`asset_id mod MPI ranks`), synchronous
observations, blocking `MPI_Allreduce`, one thread per rank, and no scan,
lookahead or persistent-team optimisation. The lookahead experiment is the
one exception: all four cells use buffered observations because the exact
lookahead implementation requires them.

## Data

The repository includes the frozen derived inputs consumed by the simulator:
the 1,480-book universe, per-book order-message distributions, pooled Hawkes
rates, queue-response policies, Value Agent controls and liquidity-cluster
assignments. It does not include raw Nasdaq ITCH messages and does not repeat
extraction or calibration. See `DATA.md` for the exact boundary between raw
data and runnable inputs.
