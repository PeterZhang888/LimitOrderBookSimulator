# Independent audit of the final inventory-stress experiment

The source evidence is the completed Seagull campaign recorded as Slurm job
45706. The external archive digest is listed in
`results/final-case-study/evidence_manifest.json`.

## Verdict

Job 45706 is complete, internally consistent and suitable for reporting.  The
corrected shared-dealer mechanism is active before the shock and operates in
the intended causal direction.  The results support a precise conclusion: an
inventory-adverse shock raises the shared dealer's aggregate exposure, reduces
its common quote scale, and causes a small, transient withdrawal of top-of-book
depth from unshocked assets.  The response is strongest during the first
2--10 seconds and is no longer distinguishable from zero by approximately
30 seconds.  The experiment does **not** support a claim of persistent
market-wide liquidity deterioration or a robust spread increase over the
30-minute post-shock horizon.

## Integrity and computational audit

- The archive SHA-256 is
  `4fe1596c539ba3e6e3153f396e85e43c13b4a1943b67299e2b94a7684d1d5973`,
  which matches the supplied checksum.
- Slurm records Job 45706 as `COMPLETED` with exit code `0:0`; elapsed time was
  2:04:23 on 16 MPI ranks and peak resident memory was 17.46 GiB.
- All 837 declared output artifacts were independently rehashed; all hashes
  matched.  Their combined size is 8,157,085,164 bytes.
- The archive contains 206 simulator executions: 80 globally coupled paths,
  80 asset-local-capacity paths, 40 shared-dealer-absent paths, four mechanism
  preflight paths and two rank-equivalence paths.
- The financial matrix contains 200 full-session paths: 20 paired seeds,
  shock and no-shock paths, and the declared treatment/control configurations.
- Rank equivalence passes exactly.  The one-rank and 16-rank executions have
  the same state hash (`0x39ca074b1c4e5b84`).  Their wall times are 14.049 s
  and 1.577 s, respectively, giving speedup 8.91 and efficiency 55.7% for this
  short preflight workload.
- The truncated/full-session prefix certificate passes, so the preflight
  behaviour is identical to the corresponding prefix of the full run.
- The 206 runs processed 25.835 billion orders and 2.049 billion trades.
  Simulator wall time sums to 6,699.0 s; job overhead was 764.0 s (10.2%).

The full-path median wall times were 35.48 s for the globally coupled dealer,
35.04 s for the asset-local-capacity control and 24.71 s with the shared dealer
absent.  Median communication fractions were 32.6%, 34.3% and 48.0%,
respectively.  The larger fraction in the absent-dealer control reflects less
local computation; it is not evidence that this control communicates more
data in absolute terms.

## Experimental identification

The empirical universe contains 1,480 books.  In every shock path, 148 books
(10%) are selected by a fixed target seed.  A quantity of 175,506 units is
requested, and the buy/sell direction in each target is chosen to worsen the
dealer's pre-shock inventory.  The paired no-shock path uses the same stochastic
seed.  The primary outcome is a difference-in-differences comparison:

1. shock minus no-shock under the globally coupled dealer; minus
2. shock minus no-shock under asset-local capacity.

The asset-local and shared-dealer-absent controls produce exactly zero
non-target difference at every recorded time.  This is expected from the
rank-independent random streams and local capacity isolation, and verifies
that non-target effects in the global treatment are transmitted by the common
inventory constraint rather than by random-number drift.

## The shared dealer is active before the shock

The earlier shutdown problem is resolved.  In the mechanism preflight, the
dealer requests two-sided quotes in 100% of books.  The minimum fraction with
two-sided resting quotes is 97.8% at capacity 800 and 98.0% at capacity 1,600.
Pre-shock utilisation is 83.7% and 83.1%, below the declared 85% headroom gate;
the corresponding quote scales are 0.326 and 0.338.  Across all 20 production
seeds, the minimum observed quote scale remains above the 0.05 floor.  The
minimum 99.73% two-sided-book statistic reported for the full matrix describes
the books overall, not shared-dealer resting coverage.

## Causal mechanism

The one-second response has the correct sign for every seed.

| Capacity per asset | Exposure increase, units (95% CI) | Utilisation increase | Quote-scale change (95% CI) | Mean shock absorption |
|---:|---:|---:|---:|---:|
| 800 | 5,384.7 [4,592.1, 6,177.2] | 0.00455 | -0.00910 [-0.01043, -0.00776] | 2.02% |
| 1,600 | 6,621.8 [6,054.7, 7,188.9] | 0.00280 | -0.00559 [-0.00607, -0.00511] | 2.52% |

The tighter capacity produces an additional quote-scale reduction of 0.00350
(paired 95% CI [0.00254, 0.00447] in absolute magnitude).  Thus the capacity
treatment changes the dealer's response as designed.  The dealer absorbs only
a small part of the market shock, but majority absorption is not required for
the causal channel: the adverse fills that it does receive are sufficient to
change the globally shared risk state.

At one second, requested depth in unshocked books has not yet changed because
the quote-scale update becomes visible at the next quote-refresh boundary.  At
five seconds, the mean reduction in requested shared depth is 3,370 units at
capacity 800 and 1,939 units at capacity 1,600.  By 30 seconds, the exposure,
quote-scale and requested-depth contrasts have confidence intervals that
include zero.

## Market-wide liquidity effects

Positive depth values below mean that the shock reduced top-of-book depth in
unshocked assets relative to both controls.

| Horizon | Capacity | Relative depth deterioration (95% CI) | Spread change, bps (95% CI) |
|---:|---:|---:|---:|
| 2 s | 800 | 0.0568% [0.0479%, 0.0656%] | 0.00033 [-0.00029, 0.00094] |
| 5 s | 800 | 0.0574% [0.0462%, 0.0685%] | 0.00135 [-0.00176, 0.00445] |
| 5 s | 1,600 | 0.0420% [0.0101%, 0.0738%] | 0.00182 [-0.00142, 0.00505] |
| 30-min mean | 800 | 0.1790% [-0.2972%, 0.6552%] | -0.00115 [-0.01431, 0.01202] |
| 30-min mean | 1,600 | 0.0108% [-0.3815%, 0.4032%] | -0.00350 [-0.01488, 0.00787] |

The early depth effect is statistically clear but economically small.  Spread
changes are not distinguishable from zero at five seconds or over the full
horizon.  The 800-versus-1,600 contrast in 30-minute mean depth is 0.168
percentage points with a paired 95% CI of [-0.496, 0.832], so the experiment
does not establish capacity-dependent persistence in the market-wide outcome.

Late maxima of the seed-averaged time path should not be described as delayed
causal peaks.  By then the paired paths have accumulated stochastic divergence,
and the seed-level 30-minute confidence intervals include zero.

## Liquidity-cluster heterogeneity

Cluster membership is not balanced: cluster sizes range from 70 to 266 books.
Cluster identifiers are categorical and are **not** an ordered scale from
illiquid to liquid.  Their baseline depth and spread characteristics must be
reported alongside any cluster label.

During seconds 2--10, the capacity-800 mean depth deterioration is positive
with a 95% interval above zero in nine of ten clusters.  The largest estimates
occur in clusters 2 (0.157%), 3 (0.132%), 5 (0.124%) and 0 (0.120%).  Under
capacity 1,600, eight clusters have positive intervals; the largest estimates
are cluster 2 (0.110%), cluster 3 (0.087%), cluster 7 (0.086%) and cluster 0
(0.073%).  This ordering is not monotone in baseline spread or depth.

Over the full 30-minute window, no cluster's depth confidence interval excludes
zero under either capacity.  Two of 20 exploratory cluster-spread intervals
exclude zero under capacity 1,600, but they have opposite signs and do not
survive a cautious multiple-comparison interpretation.  The defensible result
is therefore broad but transient early depth withdrawal, not persistent
cluster-specific contagion.

## Thesis-level interpretation

The final experiment demonstrates a causal and computational result rather
than a historical replay.  Whole-book MPI decomposition permits a
rank-independent, 1,480-book counterfactual matrix involving 25.8 billion
processed orders.  The global reduction implements a common balance-sheet
state that a sequential collection of independent books cannot represent
without an equivalent global synchronization protocol.  The experiment shows
that common inventory capacity transmits an adverse fill across otherwise
unshocked books, but the calibrated market replenishes the resulting depth
withdrawal rapidly.  This resilience result is scientifically meaningful and
should replace any claim of a sustained simulated liquidity spiral.

## Files produced by this audit

- `recomputed_results.json`: independent numerical audit.
- `tables/`: seed-level and aggregate plotting tables.
- `figures/mechanism_response.{pdf,png}`: mechanism chain.
- `figures/marketwide_depth_early.{pdf,png}`: immediate outcome.
- `figures/marketwide_depth_full.{pdf,png}`: full-horizon context.
- `figures/cluster_early_response.{pdf,png}`: cluster heterogeneity.
- `derive_plot_data.py`, `make_figures.R`, and `recompute_results.py`:
  reproducible analysis scripts.
