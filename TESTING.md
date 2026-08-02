# Verification status

This file records the release audit honestly; it is not a claim that empirical
workflows can run without their external data.

## Passed checks

- CMake configuration and compilation completed with the locally available
  Open MPI C++ toolchain.
- Every shell file passes `bash -n`.
- Every Python source file compiles with `python3 -m compileall`.
- The restored 4,228-line historical real-universe launcher passes all 28 tests
  in `test_real_universe_case_study_submission_contract.py`.
- All 13 functional artifact-resolution tests in
  `test_resolve_queue_reactive_case_artifact.py` pass.
- Twenty-six self-contained C++/executable CTest cases pass. The remaining
  local executable tests stop at startup because their empirical mark/rate
  fixtures are excluded; the sandbox also prevents Open MPI from binding a TCP
  listener, so this environment is not a cluster-MPI acceptance test.
- The R36 rank-equivalence evidence records the same state hash at 1 and 32
  ranks; the evidence and interpretation are under `docs/case-study`.

## Tests requiring external empirical fixtures

The public repository deliberately excludes raw Nasdaq ITCH archives and the
expanded per-symbol empirical distribution directories. Tests that open files
such as `data/itch_20200130_qqq/limit_buy_quantity_distribution.txt` therefore
cannot pass in this source-only collection. This is missing test data, not a
compiler or parser failure.

## Known historical source-export mismatch

Four source-contract assertions in
`QueueReactiveSubmissionSourceContractTest` describe a later queue-reactive
extension to the historical `submit_real_universe_case_study.sh`. The matching
revision of that legacy launcher was not present in any local archive, so this
release does not fabricate it or weaken the tests. The successful R36 campaign
used the separate `submit_queue_reactive_case_study.sh`, which is included with
its executed-source digest in `PROVENANCE.md`.

The successful R36 source manifest also names
`scripts/prepare_portable_queue_case.py`, but the evidence export omitted that
helper. Its expected SHA-256 and recovery command are in
`REPRODUCIBILITY.md`. Until it is recovered, describe this repository as the
complete locally available source collection, not a byte-for-byte complete R36
execution bundle.
