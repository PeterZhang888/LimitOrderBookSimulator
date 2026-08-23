# Distributed Multi-Asset Limit Order Book Simulator

> **Original data source.** The raw order-message files used to prepare the
> frozen empirical inputs were obtained from the
> [official Nasdaq TotalView--ITCH archive](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/).
> Raw ITCH files are not redistributed in this repository; access and use of
> Nasdaq data remain subject to Nasdaq's applicable terms.

This repository contains the final simulator used by the thesis. Each logical
asset is represented by one complete simulated LOB object owned by one MPI
rank. Strategic and market-maker orders retain individual price--time
priority. Anonymous background flow is represented on a moving ten-level
price band, and adjacent background quantity at the same price may be
aggregated.

OpenMP processes complete books already owned by an MPI rank. In the final
permanent-ownership mode, each book is assigned to one thread for the complete
session. In the phase-based diagnostic, a book may be processed by different
threads in successive phases, but never concurrently.

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

bash scripts/build_seagull.sh
bash scripts/submit_seagull_validation.sh
```

These two scripts provide the complete standard workflow: the first builds
and tests both executables, and the second submits representative full-session
configurations from every experiment. The submitted jobs use their
ordinary experiment names in Slurm.

The build command first checks that all frozen runtime inputs are present. It
then requires OpenMP support, builds both executables and runs the compiled
correctness tests against both builds. No raw Nasdaq message files or
calibration run are required. A successful build prints only a short progress
summary; complete compiler and test output is retained in
`build-logs/seagull-build.log`. If a build fails, the relevant final section
of that log is printed automatically. Each build also records the source
commit from which it was produced. Submission stops and requests a rebuild if
the source commit has changed or tracked source files have been modified.

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

## Validate the release on Seagull

After building a fresh clone, submit the full-session release checks with:

```bash
bash scripts/submit_seagull_validation.sh
```

The driver submits representative full-session configurations from every
experiment directory. It uses the frozen inputs and complete
23,400-second session, but does not repeat the complete rank sweeps, parameter
grids or seven-repetition performance campaigns. Validation is limited to one
or two nodes.
Check all submitted jobs with:

```bash
bash scripts/check_seagull_validation.sh
```

The checker prints `IN PROGRESS` while jobs remain queued or running. It prints
`PASS` only when every submitted job completes, the exact expected number of
simulator summaries is present, and each run has a complete boundary-metric
CSV and per-asset CSV. The OpenMP checks also require their comparison files,
and the stylised-fact check requires all 16 rank-local return panels. It prints
`FAIL` if any of these conditions is not met.

> **Scope of validation.** The release workflow reproduces simulator execution
> and simulated outputs. The derived empirical one-second return panel used in
> the final stylised-fact comparison is not included, so the complete
> empirical-versus-simulated figure cannot be regenerated from this repository
> alone. See `DATA.md`.

## Portable build

On another system with CMake, a C++20 compiler, MPI and OpenMP:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOB_REQUIRE_MPI=ON \
  -DLOB_ENABLE_OPENMP=ON \
  -DLOB_BUILD_TESTS=ON

cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The reported performance results use the documented Seagull environment and
should be reproduced with the Seagull scripts above.

## Submit a complete synthetic session

The artificial input under `examples/synthetic/` is included so the code can
run without Nasdaq data. Submit the full 10,000-book, 23,400-second session
from the repository root:

```bash
bash experiments/00_full_synthetic/submit.sh
```

The job uses 16 nodes, 256 MPI ranks, one thread per rank and one complete
session. Results are written below `results/runs/<job-id>/full_10000/`.

## Thesis experiments

Every directory under `experiments/` contains standard `compile.sh` and
`submit.sh` entry points. All experiments compile the same final simulator
source tree; their submission files change only the declared workload and
treatment settings. For example:

```bash
bash experiments/03_empirical_scaling/compile.sh
bash experiments/03_empirical_scaling/submit.sh
```

The compile file calls the central verified build, while the submit file
launches that experiment's complete formal campaign. The central release
validation remains separate: `bash scripts/submit_seagull_validation.sh`
submits representative complete-session configurations without repeating the
formal parameter grids or performance repetitions.

```text
00_full_synthetic
01_strong_scaling
02_weak_scaling
03_empirical_scaling
04_rank_ownership
05_observation_buffering
06_mpi_openmp
07_risk_collectives
08_stylised_facts
09_inventory_policy
```

The strong-scaling and MPI--OpenMP submit files are submission drivers because
their configurations need different numbers of nodes. From the repository
root, submit the complete 1--256-rank strong-scaling campaign with:

```bash
bash experiments/01_strong_scaling/submit.sh
```

Every `submit.sh` file is invoked with `bash`; it selects the correct Slurm
submission method internally. The frozen empirical universe and policies are
included under `data/empirical/`, so no cluster-specific input paths need to
be exported. A complete submission sequence is:

```bash
bash experiments/01_strong_scaling/submit.sh
bash experiments/02_weak_scaling/submit.sh
bash experiments/03_empirical_scaling/submit.sh
bash experiments/04_rank_ownership/submit.sh
bash experiments/05_observation_buffering/submit.sh
bash experiments/06_mpi_openmp/submit.sh
bash experiments/07_risk_collectives/submit.sh
bash experiments/08_stylised_facts/submit.sh
bash experiments/09_inventory_policy/submit.sh
```

These are formal full-session jobs. The main performance campaigns use seven
repetitions unless their submission file states otherwise; weak scaling uses
three, while the stylised-fact and inventory-policy simulations use their
explicitly declared paths. Experiment 06 creates its measured per-book
scheduling costs from one full preparation run before starting the timed
OpenMP comparisons; that preparation run is stored separately and is not
included in the reported comparisons.

The thesis reports two OpenMP designs. In the phase-based control, every local
phase dynamically redistributes complete books among the rank's threads. In
the final permanent-owner design, measured book costs are used to assign each
book to one thread before timing, and that ownership is retained for the
complete session. The superseded event-window-only and task-per-book
prototypes are not part of the formal experiment workflow.

The following matched campaign provides the phase-based pure-MPI versus
MPI-free OpenMP comparison at 16 physical cores:

```bash
mkdir -p slurm
sbatch experiments/06_mpi_openmp/submit_16core_mpi_openmp_pair.sh
```

It pairs 16 MPI ranks with one process times 16 threads on the same node. The
job uses seven alternating-order blocks, checks that the two layouts receive
the same physical cores, and requires exact agreement in per-asset outputs,
event counts and final accounting. Because MPI rank layouts combine
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
mkdir -p slurm
sbatch experiments/06_mpi_openmp/submit_16core_fixed_ownership_pair.sh
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
to every threaded layout at 16 and 32 total physical cores. MPI ownership
remains cyclic. Within each rank, its local books are assigned once to threads
using the common measured-cost file and retain those thread owners for the
whole session. The campaign covers
`16x1, 8x2, 4x4, 2x8, 1x16`,
and `32x1, 16x2, 8x4, 4x8, 2x16`:

```bash
bash experiments/06_mpi_openmp/submit_fixed_ownership_matrix.sh
```

The driver submits one exclusive job for each total core count. Each job uses
seven rotating-order blocks, verifies physical-core placement, requires
reproducible outputs within every layout, and compares all scientific outputs
with its pure-MPI control before accepting the timing table. Before each timed
simulation, the same MPI communicator performs 100 pairs of small blocking
reductions. A launch is rejected when the slowest-rank mean exceeds
\(2\,\mathrm{ms}\) per collective. A completed simulation is retained
regardless of its execution time. Preflight rejections remain in the result
directory and are listed in `attempts.csv`; the driver retries until each
configuration has seven completed repetitions, with a maximum of ten
preflight attempts for each required repetition. The preflight runs before
the execution timer and does not alter the simulated market state. A layout whose
maximum-to-minimum runtime ratio exceeds 1.15 is retained and labelled with a
timing-variability warning; this warning does not cause an otherwise valid
Slurm job to fail. Invalid placement, scientific disagreement, failed health
checks and incomplete repetitions remain fatal errors. The formal batch file
requests a 48-hour allocation so an unusually slow but progressing repetition
is not terminated by the former short timing limit.

To diagnose the 16-core result phase by phase, run the separate instrumented
campaign:

```bash
mkdir -p slurm
sbatch experiments/06_mpi_openmp/submit_16core_window_profile.sh
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
These diagnostic timings provide a profiled-window decomposition.  The slowest rank observed in
each window is a diagnostic proxy, not an exact reconstruction of the full
session critical path.

If a multi-node run intermittently spends several minutes inside nominally
small collectives, use the focused 32-rank diagnostic instead of repeating the
complete decomposition matrix:

```bash
bash experiments/06_mpi_openmp/submit_collective_stall_diagnostic.sh
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
selected quote multiplier 2.00; experiment 09 deliberately replaces that one
value across its declared participation sweep.

Every performance treatment writes separate boundary-metric, per-asset and
console-output files. Each job also writes `environment.txt`, containing its
compiler, MPI version, CPU description, allocation and loaded modules.

Terminal Shared Market Maker valuation excludes that dealer's own resting
orders. Remaining inventory is valued from the last external price reached on
the liquidation side; if none exists, the external opposite-side quote is
used, and the latent reference is used only when no external quote remains.
The summary reports the number of affected assets and quantities attributed to
each fallback source.

Scientific CSVs can be compared directly with:

```bash
python3 scripts/compare_scientific_outputs.py reference.csv treatment.csv
```

After a campaign finishes, collect every completed simulator row and generate
the raw and median/minimum/maximum timing tables with:

```bash
python3 scripts/summarize_results.py results/runs/<job-or-campaign-directory>
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

The executable's convenience defaults are not the thesis baseline. Formal
experiment scripts explicitly select cyclic ownership, synchronous
observations, blocking risk reduction and one thread per MPI rank. Reproduce
reported controls through the experiment scripts, not through an unqualified
direct invocation of `lob_mpi`.

## Data

The repository includes the frozen derived inputs consumed by the simulator:
the 1,480-book universe, per-book order-message distributions, pooled Hawkes
rates, queue-response policies, Value Agent controls and liquidity-cluster
assignments. It does not include raw Nasdaq ITCH messages and does not repeat
extraction or calibration. See `DATA.md` for the exact boundary between raw
data and runnable inputs.
