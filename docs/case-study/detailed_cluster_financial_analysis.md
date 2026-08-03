# Cluster-level interpretation of the inventory-stress experiment

## Statistical definition

All effects are paired across the 20 common-random-number seeds.  Confidence
intervals use the paired-seed standard error and the 19-degree-of-freedom
Student critical value, 2.093.  A positive depth effect means that the shock
reduced top-of-book depth in unshocked books relative to the paired control.

"Initial recovery" is defined as the first post-shock second followed by five
consecutive seconds for which the paired 95% confidence interval includes zero.
This definition measures the end of the initial causal episode; it does not
assert that two nonlinear stochastic paths can never diverge again later.

## Transmission and recovery of the shared-dealer mechanism

The adverse shock changes the shared dealer before it changes the books.  At
one second, exposure is higher by 5,385 units at capacity 800 and 6,622 units
at capacity 1,600.  Quote scale falls by 0.00910 and 0.00559, respectively.
Requested depth first changes at two seconds, which is consistent with the
one-second decision/refresh protocol rather than with an instantaneous book
rewrite.

| Quantity | Capacity 800 | Capacity 1,600 |
|---|---:|---:|
| Exposure/quote-scale half-decay | 10 s | 13 s |
| Exposure/quote-scale falls to 10% of initial mean | 24 s | 32 s |
| Exposure/quote-scale initial recovery | 20 s | 28 s |
| Requested-depth initial recovery | 21 s | 27 s |
| Resting-depth initial recovery | 21 s | 29 s |
| Observed market top-depth initial recovery | 14 s | 25 s |

This reveals a capacity trade-off.  The tighter dealer cuts quotes more
aggressively, so the immediate market-wide depth loss is larger.  It also
absorbs less of the shock (2.02% rather than 2.52%), accumulates a smaller
exposure difference and recovers sooner.  The looser dealer protects liquidity
more effectively in the first few seconds, but absorbs more inventory and
therefore carries a smaller disturbance for longer.

The mean market-wide depth deterioration during seconds 2--5 is 0.0575%
[0.0488%, 0.0663%] at capacity 800 and 0.0278% [0.0184%, 0.0371%] at capacity
1,600.  At capacity 800 it remains positive during seconds 6--10, then the
11--20 second interval includes zero.  At capacity 1,600 the 11--20 second
average remains positive, 0.0408% [0.0216%, 0.0600%], but the 21--30 second
interval includes zero.

## Cluster-level response

Cluster identifiers are categorical.  They should not be renamed from 0 to 9
as though they were an ordinal liquidity scale.  The table reports their actual
baseline characteristics, early response and initial recovery time.

| Cluster | Books | Baseline top depth | Baseline spread (bp) | Early effect, capacity 800 | Recovery (s) | Early effect, capacity 1,600 | Recovery (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 146 | 543 | 58.78 | 0.120% [0.094, 0.146] | 20 | 0.073% [0.057, 0.090] | 35 |
| 1 | 144 | 1,089 | 51.92 | 0.084% [0.063, 0.104] | 15 | 0.052% [0.043, 0.062] | 25 |
| 2 | 191 | 451 | 8.47 | 0.157% [0.112, 0.202] | 10 | 0.110% [0.083, 0.137] | 21 |
| 3 | 266 | 607 | 17.31 | 0.132% [0.097, 0.168] | 14 | 0.087% [0.072, 0.101] | 25 |
| 4 | 161 | 2,675 | 63.99 | 0.033% [0.025, 0.041] | 10 | 0.015% [-0.008, 0.038] | 24 |
| 5 | 196 | 709 | 6.97 | 0.124% [0.086, 0.162] | 15 | 0.070% [0.034, 0.106] | 8 |
| 6 | 145 | 1,563 | 14.68 | 0.071% [0.057, 0.084] | 12 | 0.040% [0.030, 0.049] | 16 |
| 7 | 82 | 505 | 3.03 | 0.038% [-0.040, 0.117] | 8 | 0.086% [0.028, 0.145] | 6 |
| 8 | 79 | 12,281 | 27.52 | 0.021% [0.012, 0.029] | 11 | 0.015% [0.003, 0.027] | 10 |
| 9 | 70 | 1,990 | 4.29 | 0.103% [0.042, 0.164] | 5 | -0.078% [-0.246, 0.090] | no significant initial episode |

Early effect is the mean percentage depth deterioration over seconds 2--10.
Recovery estimates can be shorter than the averaging window because they are
calculated from the second-by-second confidence sequence.

### Amplitude

The largest and most precisely estimated early withdrawals occur in clusters
2, 3, 5 and 0.  These clusters are relatively shallow.  The extremely deep
cluster 8 has the smallest response.  Across the ten cluster centroids, the
Spearman correlation between baseline top depth and early percentage depth
loss is -0.673 at capacity 800 ($p=0.033$) and -0.915 at capacity 1,600
($p=0.00020$).  Baseline spread has little relation to response magnitude
($\rho=-0.236$ and $-0.200$).

The economically relevant interpretation is that the same common percentage
reduction in dealer supply is more visible in books with small depth buffers.
Quoted spread alone does not identify this vulnerability.  A book can have a
tight spread yet remain sensitive because little quantity supports that price.

These correlations are exploratory because there are only ten cluster
centroids and clusters are constructed from several liquidity features.  They
should support, not replace, the paired treatment estimates.

### Duration

At capacity 800, median cluster recovery is 11.5 seconds, ranging from 5 seconds
(cluster 9) to 20 seconds (cluster 0).  At capacity 1,600, median recovery among
the nine clusters with an initial effect is 21 seconds, ranging from 6 seconds
(cluster 7) to 35 seconds (cluster 0).  Clusters 0 and 1, which have very wide
baseline spreads, recover most slowly under the larger capacity.  The
exploratory Spearman relation between spread and recovery is 0.75 at capacity
1,600 ($p=0.019$, nine clusters), but it is weaker at capacity 800
($\rho=0.50$, $p=0.14$).  This suggests that depth controls the amplitude of
the withdrawal, while weak replenishment associated with wide-spread regimes
may influence its duration.  The latter interpretation is tentative because
recovery is a threshold-based statistic and cluster 9 has no capacity-1,600
episode to time.

## Spread, prices and persistence

The primary transmission channel is quantity, not price.  During seconds 2--5,
the market-wide spread change is 0.00041 bp at capacity 800 and 0.00138 bp at
capacity 1,600; both intervals include zero.  Capacity 800 shows a small
0.00251 bp increase during seconds 6--10, but later spread estimates change
sign.  No cluster has a robust early spread increase across the two capacities.

Over the entire 30-minute horizon, mean depth and spread effects include zero
for both capacities.  None of the 20 cluster-by-capacity depth intervals
excludes zero.  Two exploratory cluster spread intervals exclude zero at
capacity 1,600, but they have opposite signs.  This is incompatible with a
coherent persistent-contagion interpretation.

An isolated capacity-800 depth estimate over seconds 121--300 is positive by a
narrow margin, and an isolated spread estimate over seconds 301--600 is also
non-zero.  Neither is accompanied by a contemporaneous exposure or quote-scale
effect, the signs are not stable across adjacent windows, and the full-horizon
paired result is null.  Given the number of inspected times and clusters, these
late observations should be treated as stochastic path divergence rather than
as delayed dealer-mediated contagion.

## Financial conclusion

The experiment identifies a short-lived common-liquidity externality.  A shock
to 10% of the books worsens a shared supplier's portfolio inventory; the common
risk rule then reduces its quantities in the other 90%.  Shallow books bear the
largest proportional loss.  The market does not enter a persistent simulated
liquidity spiral: background order flow and local suppliers replenish the lost
depth within tens of seconds, and spreads do not widen materially.

The strongest thesis claim is therefore not that all shocks create systemic
collapse.  It is that the MPI model can isolate, measure and time a causal
cross-asset inventory channel across 1,480 books, and that under the calibrated
conditions this channel is real but rapidly mean-reverting.
