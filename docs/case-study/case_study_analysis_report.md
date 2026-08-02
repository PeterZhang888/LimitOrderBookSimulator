# R36 cluster and case-study audit

The 40-path matrix and fixed 15-symbol target mask were verified. The target mask covers all ten clusters. The corrected asset-summary spread conversion reconciles to the recorded market-wide series: 45.203195 versus 45.251398 bps in the reference path.

All non-target symbol effects are exactly zero. The largest absolute shock-minus-control difference is 0.0 bps for spread and 0.0 shares for top depth. Consequently, every cluster-level cross-book effect and every global-minus-uncoupled difference-in-differences estimate is exactly zero.

This is a structural null. The globally constrained maker became permanently inactive after 2.41 minutes at capacity 25 and 23.62 minutes at capacity 100, before the shock at 195 minutes. Cluster-specific contagion cannot be ranked because the common channel was absent.

The direct target response is identifiable only in aggregate over the 15 targets. The archived per-second files do not retain target-by-cluster trajectories, and most clusters contain only one or two targeted books. Full-session target-by-cluster summaries are included in the JSON as descriptive diagnostics, not causal cluster estimates.
