# Source and data provenance

## Source code

The production implementation is maintained in this repository. An audit of
the available source tree found no vendored or visibly copied third-party source
code. Standard-library use, MPI calls, and implementations of published file
formats or mathematical models do not by themselves constitute copied source.

If externally authored code is incorporated later, the affected file must name
the original author, source URL, licence and local modifications. The same
material must be recorded in `THIRD_PARTY_NOTICES.md`.

## Specifications and model inputs

- The binary parser implements the publicly documented Nasdaq TotalView--ITCH
  5.0 message layout; no Nasdaq implementation source is included.
- The stochastic order-flow model uses Hawkes-process and queue-reactive model
  concepts described and cited in the accompanying academic work.
- The market-making mechanisms implement inventory- and capacity-dependent
  policies described by the model specification.
- `config/qqq_reduced_basket_weights_20190930.csv` contains derived weights;
  its SEC filing source is recorded in each row and in the accompanying method
  note.

These sources require scholarly citation but do not imply source-code
authorship by the cited researchers or institutions.

## Empirical data

Raw Nasdaq ITCH archives are external and are not redistributed. Derived data
directories remain subject to the terms governing their source data. The
repository contains only small configurations, cohort identifiers, code and
compact result summaries. See `DATA.md` for the required local layout.

## Result evidence

`results/final-case-study/` contains the compact outputs retained from the
completed campaign. `docs/case-study/analysis_manifest.json` records the source
archive digest and generated analysis products. Absolute cluster paths are not
required to interpret the published summary statistics.

## Dependencies

Compiler, C++ standard-library, CMake, MPI, Python and optional Apple Metal
components are supplied by the execution environment and remain governed by
their own licences. See `THIRD_PARTY_NOTICES.md`.
