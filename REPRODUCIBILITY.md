# Reproducibility notes

## What is preserved

- the full available C++/MPI implementation and research workflow;
- the exact five R36 files included in the successful job-45498 evidence export,
  identified by their pre-attribution hashes in `PROVENANCE.md`;
- tests, Slurm launchers and deterministic source-manifest tooling;
- an audited compact R36 analysis and thesis-ready figures under
  `docs/case-study`.

## What is external

- raw and derived ITCH data;
- pooled 2019 per-symbol distributions;
- the selected cluster-policy directory;
- the hash-bound calibration and development-validation handoffs;
- the exact `prepare_portable_queue_case.py` helper omitted by the downloaded
  R36 evidence archive.

The original helper can be recovered on Seagull with:

```bash
scp -o ProxyJump=czhang4@rsync.tchpc.tcd.ie \
  czhang4@seagull.tchpc.tcd.ie:/home/users/mschpc/2025/czhang4/coupled_lob_r36_case_integrated_20260802/scripts/prepare_portable_queue_case.py \
  ./scripts/
```

After recovering it, first verify the untouched executed-source bytes:

```bash
printf '%s  %s\n' \
  b5335de40d5d5ac7b5f48e3abf9372efb1a389360dec480b0f5fc07122ae26fd \
  scripts/prepare_portable_queue_case.py | sha256sum -c -
```

Only after that verification, add the project provenance notice after its
shebang and regenerate the manifest:

```bash
python3 scripts/generate_source_manifest.py \
  --root . --output SOURCE_MANIFEST.sha256
sha256sum -c SOURCE_MANIFEST.sha256
```

## Local verification without empirical data

Configure and build the source, then run unit tests that do not open empirical
distribution files. The complete CTest suite requires the omitted derived data.
Python syntax and unit tests with self-contained temporary fixtures can be run
directly from the repository.

## Scientific result boundary

Job 45498 is evidence for deterministic rank equivalence and the execution of
the predeclared case matrix. It is not evidence for cross-asset contagion: the
global shared market maker's quote scale was zero before the shock. The detailed
audit is preserved in `docs/case-study/case_study_analysis_report.md`.
