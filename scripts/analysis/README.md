# Inventory-stress analysis

These scripts reproduce the independent audit, time-resolved summaries and
figures from a complete `queue_case_JOBID` result directory.

Python requirements are Python 3.10 or later, NumPy and pandas. Figure
generation additionally requires R and ggplot2.

```bash
RESULT_ROOT=/absolute/path/to/results/seagull/queue_case_JOBID
OUTPUT_ROOT=/absolute/path/to/analysis

python3 scripts/analysis/audit_inventory_stress_results.py \
  "$RESULT_ROOT" \
  --output "$OUTPUT_ROOT/independent_audit.json"

python3 scripts/analysis/derive_inventory_stress_tables.py \
  "$RESULT_ROOT" \
  "$OUTPUT_ROOT/tables"

python3 scripts/analysis/analyze_inventory_stress_dynamics.py \
  "$RESULT_ROOT" \
  "$OUTPUT_ROOT/dynamics"

Rscript scripts/analysis/make_inventory_stress_figures.R \
  "$OUTPUT_ROOT/tables" \
  "$OUTPUT_ROOT/figures"

Rscript scripts/analysis/make_inventory_stress_cluster_figures.R \
  "$OUTPUT_ROOT/tables/cluster_early_response_summary.csv" \
  "$OUTPUT_ROOT/dynamics" \
  "$OUTPUT_ROOT/figures"
```

The audit verifies every artifact listed in `case_job_completion.json` before
computing financial summaries. The detailed recovery definition is the first
post-shock second followed by five consecutive seconds whose paired 95%
confidence intervals include zero. It measures the end of the initial causal
episode, not permanent convergence of stochastic paths.
