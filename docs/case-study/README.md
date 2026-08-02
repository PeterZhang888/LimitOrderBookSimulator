# R36 case-study thesis package

Use `case_study_chapter.tex` as the main thesis section. It requires the
`booktabs`, `graphicx` and `float` packages and the three PDF figures in this
directory. `case_study_cluster_appendix.tex` is optional: it reports sparse,
full-session target-by-cluster diagnostics and explicitly does not treat them
as causal cluster estimates. `case_study_preview.pdf` is a compiled visual
check of both files.

The independent audit found that the archived auxiliary cluster summary used
the wrong scale when labelling spread as basis points. The corrected conversion
is

```text
mean_spread_ticks * 1,000,000 / fundamental_price_ticks
```

It reconstructs a reference non-target mean of 45.203195 bps, compared with
45.251398 bps in the recorded market-wide time series. The discrepancy is only
the expected difference between averaging asset summaries and the per-second
aggregate. The primary per-second shock effects were already recorded in basis
points and are not changed by this correction.

The principal scientific conclusion is a mechanism diagnostic: the shared
market maker reached an absorbing zero-quote state before the shock. Therefore,
the exact zero response in all non-target clusters is not evidence of market
resilience or the absence of contagion under an active common dealer.
