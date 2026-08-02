# Source provenance

## Authorship disclosure

The repository is project code developed for Peter Zhang's master's thesis with
substantial OpenAI ChatGPT/Codex assistance in design, implementation,
debugging, testing, documentation and analysis. This disclosure should be kept
when the repository is made public and should be reconciled with the
university's academic-integrity and generative-AI policies.

Source headers use the following wording:

> Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.

For Python and shell programs the interpreter shebang must be line 1, so the
notice is line 2. In C++, Objective-C++ and CMake files it is line 1. Data formats
such as CSV and JSON do not permit comments reliably; their provenance is
recorded in manifests and documentation instead.

## External-code audit

The available project trees and the final R36 evidence export were searched for
copyright notices, licence notices, repository URLs and statements such as
"copied", "adapted" or "derived". No vendored or visibly copied third-party
source code was identified. Standard-library imports, MPI calls and an
implementation of a published file format or mathematical model are not, by
themselves, copied source code.

This is an evidence-based audit, not a legal guarantee or a global source-code
similarity proof. If a contributor later introduces externally authored code,
the first permissible comment line must identify the author, source URL,
licence and local modifications, and the material must also be added to
`THIRD_PARTY_NOTICES.md`.

## Research and specification influences

- The binary parser follows the publicly documented Nasdaq TotalView--ITCH 5.0
  message layout. No Nasdaq implementation source is included.
- The stochastic order-flow implementation uses Hawkes-process concepts.
- The market-making and shared-inventory mechanisms are research-model
  implementations informed by the academic literature discussed in the thesis.
- `config/qqq_reduced_basket_weights_20190930.csv` contains derived weights with
  its SEC filing source recorded in every row and in the accompanying methodology.

These influences require scholarly citation, but they do not establish source
code authorship by the cited researchers or institutions.

## Final-source lineage

The final source was integrated onto the existing public-project lineage at
commit `0c14558`. All files unique to that earlier history, including the
`Draft/` tree, legacy header layout and small empirical inputs, are retained.
The only path relocation is the empty tracked file `include/agents`, moved to
`legacy-layout/include-agents-placeholder.txt` because its filename prevented
creation of the production `include/agents/` directory. Git records this as a
rename rather than a deletion.

The GitHub collection was assembled from the complete R33 source tree and the
five scientifically material files exported with successful R36 job 45498. The
R36 source archive has SHA-256
`8355e6a788a9f0feeae14cdd2b31358eee7210c7100acb7be378f781d62682d9`.
Before attribution comments were added, the exact R36 file digests were:

| Path | SHA-256 |
|---|---|
| `src/fragmented_mpi_main.cpp` | `8af0559ac48d467c573d0d15a383ce2d8b46f9b65f98fb3da8d344e674392a73` |
| `src/simulation/FragmentedMpiSimulator.cpp` | `c2549992d5e196ea480be55d3b6424ead3f588108fe2360b966cfcf9d1effae9` |
| `scripts/run_fragmented_mpi_experiments.py` | `10ff2773226dd157317c0ea48efa9064d9fc60c315f28624bff48a2e78de75a1` |
| `scripts/analyze_cluster_liquidity_heterogeneity.py` | `29b6b536f99b7d0ac1e14192303c1879b75dd7a37a90ca1cb112ed80422bdbd5` |
| `submit_queue_reactive_case_study.sh` | `8d25c6b82e55c2fbaaa5cce5a9f156c0947589ca2212e573f20e6df72829217e` |

The newly generated `SOURCE_MANIFEST.sha256` binds the GitHub-ready tree after
documentation and attribution comments were added.

The R33 copy of `submit_real_universe_case_study.sh` was itself a truncated
terminal/text export. The complete 4,228-line historical launcher was recovered
from `v19_complete_evidence_cal45321_pack45334.tar.gz`; its pristine SHA-256 was
`31bbf78ff396a67b5714ea611cbc272bf79600fd72e669ca2f16ae9024244439`
before the project-attribution comment was added. This launcher is retained for
workflow history. The successful final campaign used
`submit_queue_reactive_case_study.sh`.

## Known source-export gap

The R36 result archive contains the successful launcher and its source manifest,
but not `scripts/prepare_portable_queue_case.py`. Its executed-source SHA-256 is
`b5335de40d5d5ac7b5f48e3abf9372efb1a389360dec480b0f5fc07122ae26fd`.
The exact helper should be recovered from the original Seagull R36 project
directory before claiming a complete byte-for-byte source release. No
substitute is silently presented as the executed helper.
