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

The strong-scaling and MPI--OpenMP files are submission drivers because their
configurations need different numbers of nodes. From the repository root,
submit the complete 1--256-rank strong-scaling campaign with:

```bash
bash experiments/01_strong_scaling/submit_seagull.sh
```

Experiments 01 and 07 are invoked with `bash`; the remaining experiment files
are passed directly to `sbatch`. The frozen empirical universe and policies
are included under `data/empirical/`, so no cluster-specific input paths need
to be exported. A complete submission sequence is:

```bash
mkdir -p slurm results/seagull
bash experiments/01_strong_scaling/submit_seagull.sh
sbatch experiments/02_weak_scaling/submit_seagull.sh
sbatch experiments/03_empirical_scaling/submit_seagull.sh
sbatch experiments/04_rank_ownership/submit_seagull.sh
sbatch experiments/05_observation_buffering/submit_seagull.sh
sbatch experiments/06_fused_metric_scans/submit_seagull.sh
bash experiments/07_mpi_openmp/submit_seagull.sh
sbatch experiments/08_risk_collectives/submit_seagull.sh
sbatch experiments/09_stylised_facts/submit_seagull.sh
sbatch experiments/10_inventory_policy/submit_seagull.sh
```

These are formal full-session jobs. The performance experiments use seven
repetitions by default. Experiment 07 creates its measured per-book scheduling
costs from one full preparation run before starting the timed OpenMP
comparisons; that preparation run is stored separately and is not included in
the reported comparisons.

The thesis reports two OpenMP designs. In the phase-based control, every local
phase dynamically redistributes complete books among the rank's threads. In
the final permanent-owner design, measured book costs are used to assign each
book to one thread before timing, and that ownership is retained for the
complete session. The superseded event-window-only and task-per-book
prototypes are not part of the formal experiment workflow.

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

The permanent-book-ownership campaign tests the alternative OpenMP design in
which scheduling occurs once rather than once per phase. A full preparation
run first measures the work of every one of the 1,480 books. The books are
then sorted by measured cost and assigned, one at a time, to the currently
least-loaded OpenMP thread. This produces 16 load-balanced thread buckets.
The buckets are frozen before the timed runs: the same thread initializes,
processes events for, and applies every agent and observation phase to each of
its books throughout the complete 23,400-second session. One OpenMP team
remains alive for the session; this mode uses neither repeated
`schedule(dynamic,1)` assignment nor one OpenMP task per book.

```bash
mkdir -p slurm results/seagull
sbatch experiments/07_mpi_openmp/submit_16core_fixed_ownership_pair.sh
```

The job compares the fixed-owner one-process/16-thread configuration with the
16-rank/one-thread cyclic MPI control on the same node and physical cores in
seven alternating blocks. It records the permanent book-to-thread mapping
and rejects a run if that mapping changes, if a book has no unique owner, or
if the scientific outputs diverge. Because the OpenMP treatment also replaces
cyclic ownership with measured-cost thread assignment, this is a comparison
of two complete execution configurations; it does not isolate the programming
model from the ownership policy.

The complete permanent-owner decomposition campaign applies the same design
to every threaded layout at 16, 32 and 64 total physical cores. MPI ownership
remains cyclic. Within each rank, its local books are assigned once to threads
using the common measured-cost file and retain those thread owners for the
whole session. The campaign covers
`16x1, 8x2, 4x4, 2x8, 1x16`,
`32x1, 16x2, 8x4, 4x8, 2x16`, and
`64x1, 32x2, 16x4, 8x8, 4x16`:

```bash
mkdir -p slurm results/seagull
bash experiments/07_mpi_openmp/submit_fixed_ownership_matrix.sh
```

The driver submits one exclusive job for each total core count. Each job uses
seven rotating-order blocks, verifies physical-core placement, requires
reproducible outputs within every layout, and compares all scientific outputs
with its pure-MPI control before accepting the timing table.

To diagnose the 16-core result phase by phase, run the separate instrumented
campaign:

```bash
mkdir -p slurm results/seagull
sbatch experiments/07_mpi_openmp/submit_16core_window_profile.sh
```

It compares 16 one-thread MPI ranks with one 16-thread MPI-free OpenMP process
in three alternating blocks on the same 16 physical cores.  For every
one-second simulated interval it records event processing, the local exposure
scan, risk synchronization, asset-moment and global-metric scans, and each
agent phase.  Rows are kept in memory and written only after the simulated
session; no profiling collective or file write is inserted inside a window.
An OpenMP phase includes its useful work, scheduling and implicit end-of-loop
barrier; it is not a barrier-only measurement.  The reported MPI collective
time likewise includes rank-arrival waiting and MPI progress, not just data
movement.
These diagnostic timings provide a profiled-window decomposition and do not
replace the uninstrumented performance results.  The slowest rank observed in
each window is a diagnostic proxy, not an exact reconstruction of the full
session critical path.

If a multi-node run intermittently spends several minutes inside nominally
small collectives, use the focused 32-rank diagnostic instead of repeating the
complete decomposition matrix:

```bash
mkdir -p slurm results/seagull
bash experiments/07_mpi_openmp/submit_collective_stall_diagnostic.sh
```

It repeats only the 32-rank/one-thread layout and stops after capturing both a
normal run and a run whose execution time reaches 120 seconds, or after twelve
attempts. For each repetition it records the CPU mask of the actual process,
per-rank work before every risk reduction, experienced collective time, and
the full per-window phase decomposition. The resulting
`boundary_summary.csv` separates variation in rank-local arrival work from
time experienced inside the collective. The diagnostic does not enable an
extra boundary barrier and does not alter the financial model.
If only the post-run analysis fails, preserve the captured result directory
and submit `analyze_collective_stall.sbatch` with the original job number; the
simulation is not repeated.

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
