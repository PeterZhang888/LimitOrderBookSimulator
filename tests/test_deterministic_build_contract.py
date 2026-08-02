#!/usr/bin/env python3
# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
"""Guards for byte-identical calibration and final-case MPI builds."""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CALIBRATION_SCRIPT = ROOT / "submit_cluster_value_agent_calibration.sh"
CASE_SCRIPT = ROOT / "submit_real_universe_case_study.sh"
CONTRACT = ROOT / "scripts" / "seagull_deterministic_build.sh"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DeterministicBuildContractTest(unittest.TestCase):
    def test_both_jobs_call_only_the_shared_configure_function(self) -> None:
        for script in (CALIBRATION_SCRIPT, CASE_SCRIPT):
            source = script.read_text(encoding="utf-8")
            self.assertIn(
                'source "${DETERMINISTIC_BUILD_CONTRACT}"', source
            )
            self.assertEqual(
                source.count("lob_deterministic_configure_and_build"), 1
            )
            self.assertNotIn('cmake -S "${PROJECT_DIR}"', source)

    def test_module_toolchains_are_identical(self) -> None:
        pattern = re.compile(r'^SEAGULL_MODULES="\$\{SEAGULL_MODULES:-([^}]*)\}"$', re.M)
        calibration = pattern.search(CALIBRATION_SCRIPT.read_text(encoding="utf-8"))
        case = pattern.search(CASE_SCRIPT.read_text(encoding="utf-8"))
        self.assertIsNotNone(calibration)
        self.assertIsNotNone(case)
        assert calibration is not None and case is not None
        self.assertEqual(calibration.group(1), case.group(1))

    def test_contract_contains_all_binary_affecting_cmake_controls(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        for token in (
            'LOB_DETERMINISTIC_BUILD_CONTRACT_VERSION="seagull_release_mpi_v1"',
            '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_CXX_COMPILER="${compiler_path}"',
            '-DCMAKE_CXX_COMPILER_LAUNCHER=',
            '-DCMAKE_BUILD_RPATH="${mpi_lib_dir}"',
            '-DCMAKE_INSTALL_RPATH="${mpi_lib_dir}"',
            '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF',
            '-DLOB_REQUIRE_MPI=ON',
            '-DLOB_BUILD_TESTS=ON',
        ):
            self.assertIn(token, source)
        for variable in ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS"):
            self.assertIn(variable, source)

    def test_build_contract_is_part_of_every_workflow_hash(self) -> None:
        pool = load_module(
            "pool_build_contract_test",
            ROOT / "scripts" / "pool_multiday_empirical_universe.py",
        )
        calibration = load_module(
            "calibration_build_contract_test",
            ROOT / "scripts" / "calibrate_cluster_value_agents.py",
        )
        relative = "scripts/seagull_deterministic_build.sh"
        self.assertIn(relative, pool.WORKFLOW_SEMANTICS_FILES)
        self.assertIn(relative, calibration.WORKFLOW_SEMANTICS_FILES)
        self.assertIn(
            f'"{relative}"', CASE_SCRIPT.read_text(encoding="utf-8")
        )

    def test_exact_cohort_contract_is_part_of_every_workflow_hash(self) -> None:
        pool = load_module(
            "pool_cohort_contract_test",
            ROOT / "scripts" / "pool_multiday_empirical_universe.py",
        )
        calibration = load_module(
            "calibration_cohort_contract_test",
            ROOT / "scripts" / "calibrate_cluster_value_agents.py",
        )
        case_source = CASE_SCRIPT.read_text(encoding="utf-8")
        for relative in (
            "scripts/certification_cohort.py",
            "config/certification_symbols_1480.txt",
            "config/certification_symbols_1480_origin.json",
        ):
            self.assertIn(relative, pool.WORKFLOW_SEMANTICS_FILES)
            self.assertIn(relative, calibration.WORKFLOW_SEMANTICS_FILES)
            self.assertIn(f'"{relative}"', case_source)

    def test_calibration_submission_is_five_day_fail_closed(self) -> None:
        source = CALIBRATION_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            ': "${MULTIDAY_POOLING_PROVENANCE:?provide schema-v7 five-day '
            'pooling_provenance.json}"',
            source,
        )
        self.assertIn(
            ': "${POOLING_PRODUCER_PROJECT_ROOT:?provide the source-tree root '
            'that produced MULTIDAY_POOLING_PROVENANCE}"',
            source,
        )
        self.assertIn(
            '--pooling-provenance "${MULTIDAY_POOLING_PROVENANCE}"', source
        )
        self.assertIn(
            '--pooling-producer-project-root '
            '"${POOLING_PRODUCER_PROJECT_ROOT}"',
            source,
        )
        self.assertIn(
            '--pooled-training-universe-config "${MULTIDAY_POOLED_CONFIG}"',
            source,
        )
        self.assertIn("if eligible_count != 1480:", source)
        self.assertIn(
            "certified calibration requires exactly 1480 common symbols",
            source,
        )
        for obsolete in (
            "--training-universe-config",
            "--training-target-root",
        ):
            self.assertNotIn(obsolete, source)
        for obsolete_assignment in (
            "TRAINING_UNIVERSE_CONFIG",
            "TRAINING_TARGET_ROOT",
            "HELDOUT_OPENING_SOURCE_CONFIG",
        ):
            self.assertNotRegex(
                source, rf"(?m)^{obsolete_assignment}="
            )

    def test_calibration_runs_complete_single_process_ctest_contract(self) -> None:
        source = CALIBRATION_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('--no-tests=error', source)
        self.assertIn("-E '_mpi_[2-9][0-9]*$'", source)
        self.assertNotIn(
            "fragmented_mpi_smoke_local|fragmented_calibration_output|"
            "itch_extractor_python",
            source,
        )

    def test_calibration_result_is_published_atomically(self) -> None:
        source = CALIBRATION_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'CALIBRATION_RESULT_PATH="${RESULT_DIR}/calibration_result.json"',
            source,
        )
        self.assertIn(
            'mktemp "${RESULT_DIR}/.calibration_result.${SLURM_JOB_ID}.XXXXXX"',
            source,
        )
        self.assertIn(
            '"${CALIBRATION_ARGS[@]}" | tee "${CALIBRATION_RESULT_TMP}"',
            source,
        )
        self.assertIn(
            'mv -- "${CALIBRATION_RESULT_TMP}" "${CALIBRATION_RESULT_PATH}"',
            source,
        )
        self.assertNotIn(
            '| tee "${RESULT_DIR}/calibration_result.json"', source
        )


if __name__ == "__main__":
    unittest.main()
