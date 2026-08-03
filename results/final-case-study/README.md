# Final inventory-stress evidence

This directory contains compact evidence from the completed 1,480-book
inventory-stress campaign. The full result archive is not committed because it
is 1.80 GB compressed and 8.16 GB after expansion.

`raw-summary/` contains files written by the production workflow, including
the mechanism and prefix certificates, execution summaries and the complete
hash-bound artifact index. `derived/` contains independently recomputed
time-resolved and cluster-level summaries. The scripts used to produce the
derived files are in `scripts/analysis/`.

The result should be interpreted as a short-lived common-liquidity effect. The
inventory-adverse intervention increases the shared dealer's exposure and
reduces its common quote scale in every seed. Top-of-book depth in unshocked
assets declines by approximately 0.04--0.06% during the first few seconds and
recovers within tens of seconds. The experiment does not establish a
persistent 30-minute depth or spread effect.

The external archive is identified by `evidence_manifest.json`. To verify a
downloaded copy:

```bash
shasum -a 256 final_inventory_stress_45706_complete_evidence.tar.gz
```

The digest must equal the value in the manifest before any reanalysis.
