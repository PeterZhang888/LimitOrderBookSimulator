# Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
cmake_minimum_required(VERSION 3.20)
project(thesis_dis_smc_abc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LOB_REQUIRE_MPI "Fail configuration when a real MPI implementation is unavailable" OFF)
option(LOB_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(LOB_BUILD_TESTS "Build correctness tests" ON)

if(LOB_REQUIRE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
else()
    find_package(MPI QUIET COMPONENTS CXX)
endif()

add_library(dlob_core
    src/agents/AgentPopulation.cpp
    src/agents/EtfArbitrageAgent.cpp
    src/agents/FundamentalValueAgent.cpp
    src/agents/InformedTraderAgent.cpp
    src/agents/LargeInstitutionalAgent.cpp
    src/agents/MomentumAgent.cpp
    src/agents/PassiveMarketMakerAgent.cpp
    src/agents/SharedMarketMakerAgent.cpp
    src/calibration/CalibrationParameters.cpp
    src/calibration/EmpiricalTargets.cpp
    src/calibration/SimulationRecorder.cpp
    src/calibration/SmcAbc.cpp
    src/common/EmpiricalDistribution.cpp
    src/common/PerformanceMetrics.cpp
    src/common/RunConfig.cpp
    src/exchange/BackgroundHawkesAgent.cpp
    src/exchange/DistributedLimitOrderBook.cpp
    src/mpi/AsyncExchangeLoop.cpp
    src/mpi/EventDrivenExchangeLoop.cpp
    src/mpi/MpiTransport.cpp
    src/mpi/SharedMarketSnapshot.cpp
    src/simulation/DistributedSimulator.cpp
    src/simulation/BatchedMpiMultiAssetSimulator.cpp
    src/simulation/ExactMpiMultiAssetSimulator.cpp
    src/simulation/FragmentedMpiSimulator.cpp
    src/simulation/MultiAssetConfiguration.cpp
    src/simulation/QueueReactiveBackgroundPolicy.cpp
    src/simulation/SequentialMultiAssetSimulator.cpp
)

target_include_directories(dlob_core PUBLIC include)
target_compile_options(dlob_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(MPI_CXX_FOUND)
    target_link_libraries(dlob_core PUBLIC MPI::MPI_CXX)
    target_compile_definitions(dlob_core PUBLIC LOB_REAL_MPI_BUILD=1)
else()
    if(LOB_REQUIRE_MPI)
        message(FATAL_ERROR "LOB_REQUIRE_MPI=ON but MPI C++ was not found")
    endif()
    message(WARNING "MPI was not found. Building single-process correctness mode only.")
    target_compile_definitions(dlob_core PUBLIC LOB_FORCE_MPI_STUB=1)
endif()

if(LOB_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dlob_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(dlob_core PUBLIC -fsanitize=address,undefined)
endif()

add_executable(distributed_lob src/main.cpp)
target_link_libraries(distributed_lob PRIVATE dlob_core)
target_compile_options(distributed_lob PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_executable(smc_abc_calibrate src/smc_abc_main.cpp)
target_link_libraries(smc_abc_calibrate PRIVATE dlob_core)
target_compile_options(smc_abc_calibrate PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_executable(eligibility_evaluate src/eligibility_main.cpp)
target_link_libraries(eligibility_evaluate PRIVATE dlob_core)
target_compile_options(eligibility_evaluate PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_executable(sequential_multi_asset_lob src/sequential_multi_asset_main.cpp)
target_link_libraries(sequential_multi_asset_lob PRIVATE dlob_core)
target_compile_options(sequential_multi_asset_lob PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_executable(exact_mpi_multi_asset_lob src/exact_mpi_multi_asset_main.cpp)
target_link_libraries(exact_mpi_multi_asset_lob PRIVATE dlob_core)
target_compile_options(exact_mpi_multi_asset_lob PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_executable(batched_mpi_multi_asset_lob src/batched_mpi_multi_asset_main.cpp)
target_link_libraries(batched_mpi_multi_asset_lob PRIVATE dlob_core)
target_compile_options(batched_mpi_multi_asset_lob PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_executable(fragmented_mpi_lob src/fragmented_mpi_main.cpp)
target_link_libraries(fragmented_mpi_lob PRIVATE dlob_core)
target_compile_options(fragmented_mpi_lob PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

# Apple-GPU feasibility benchmark.  This intentionally benchmarks the
# data-parallel agent-decision phase against a bit-exact single-thread CPU
# reference; it is not presented as a GPU port of the sequential matcher.
if(APPLE)
    enable_language(OBJCXX)
    add_executable(gpu_agent_benchmark src/gpu_agent_benchmark.mm)
    set_target_properties(gpu_agent_benchmark PROPERTIES
        OBJCXX_STANDARD 20
        OBJCXX_STANDARD_REQUIRED ON
        OBJCXX_EXTENSIONS OFF
    )
    target_compile_options(gpu_agent_benchmark PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
    target_link_libraries(gpu_agent_benchmark PRIVATE
        "-framework Foundation"
        "-framework Metal"
    )
endif()

if(LOB_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_executable(test_event_ordering tests/test_event_ordering.cpp)
        target_link_libraries(test_event_ordering PRIVATE dlob_core)
        add_test(NAME event_ordering COMMAND test_event_ordering)

        add_executable(test_lob_reports tests/test_lob_reports.cpp)
        target_link_libraries(test_lob_reports PRIVATE dlob_core)
        add_test(NAME lob_reports COMMAND test_lob_reports)

        add_executable(test_population_partition tests/test_population_partition.cpp)
        target_link_libraries(test_population_partition PRIVATE dlob_core)
        add_test(NAME population_partition COMMAND test_population_partition)

        add_executable(test_calibration_parameters tests/test_calibration_parameters.cpp)
        target_link_libraries(test_calibration_parameters PRIVATE dlob_core)
        add_test(NAME calibration_parameters COMMAND test_calibration_parameters)

        add_executable(test_smc_math tests/test_smc_math.cpp)
        target_link_libraries(test_smc_math PRIVATE dlob_core)
        add_test(NAME smc_math COMMAND test_smc_math)

        add_executable(test_empirical_targets tests/test_empirical_targets.cpp)
        target_link_libraries(test_empirical_targets PRIVATE dlob_core)
        add_test(NAME empirical_targets COMMAND test_empirical_targets)

        add_executable(test_event_driven_population tests/test_event_driven_population.cpp)
        target_link_libraries(test_event_driven_population PRIVATE dlob_core)
        add_test(NAME event_driven_population COMMAND test_event_driven_population)

        add_executable(test_shared_market_snapshot tests/test_shared_market_snapshot.cpp)
        target_link_libraries(test_shared_market_snapshot PRIVATE dlob_core)
        add_test(NAME shared_market_snapshot COMMAND test_shared_market_snapshot)

        add_executable(test_trade_execution tests/test_trade_execution.cpp)
        target_link_libraries(test_trade_execution PRIVATE dlob_core)
        add_test(NAME trade_execution COMMAND test_trade_execution)

        add_executable(test_cancel_nearest_level tests/test_cancel_nearest_level.cpp)
        target_link_libraries(test_cancel_nearest_level PRIVATE dlob_core)
        add_test(NAME cancel_nearest_level COMMAND test_cancel_nearest_level)

        add_executable(test_background_quote_improvement
            tests/test_background_quote_improvement.cpp)
        target_link_libraries(test_background_quote_improvement PRIVATE dlob_core)
        add_test(NAME background_quote_improvement
                 COMMAND test_background_quote_improvement)

        add_executable(test_background_hawkes_stream
            tests/test_background_hawkes_stream.cpp)
        target_link_libraries(test_background_hawkes_stream PRIVATE dlob_core)
        add_test(NAME background_hawkes_stream
                 COMMAND test_background_hawkes_stream)

        add_executable(test_queue_reactive_background_policy
            tests/test_queue_reactive_background_policy.cpp)
        target_link_libraries(test_queue_reactive_background_policy PRIVATE dlob_core)
        add_test(NAME queue_reactive_background_policy
                 COMMAND test_queue_reactive_background_policy)

        add_executable(test_trade_tape_hash tests/test_trade_tape_hash.cpp)
        target_link_libraries(test_trade_tape_hash PRIVATE dlob_core)
        add_test(NAME trade_tape_hash COMMAND test_trade_tape_hash)

        add_executable(test_shared_market_maker tests/test_shared_market_maker.cpp)
        target_link_libraries(test_shared_market_maker PRIVATE dlob_core)
        add_test(NAME shared_market_maker COMMAND test_shared_market_maker)

        add_executable(test_multiasset_ordering tests/test_multiasset_ordering.cpp)
        target_link_libraries(test_multiasset_ordering PRIVATE dlob_core)
        add_test(NAME multiasset_ordering COMMAND test_multiasset_ordering)

        add_executable(test_sequential_multiasset_determinism
            tests/test_sequential_multiasset_determinism.cpp)
        target_link_libraries(test_sequential_multiasset_determinism PRIVATE dlob_core)
        add_test(NAME sequential_multiasset_determinism
                 COMMAND test_sequential_multiasset_determinism)
        set_tests_properties(sequential_multiasset_determinism PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )

        add_executable(test_exact_mpi_equivalence
            tests/test_exact_mpi_equivalence.cpp)
        target_link_libraries(test_exact_mpi_equivalence PRIVATE dlob_core)
        add_test(NAME exact_mpi_equivalence_local
                 COMMAND test_exact_mpi_equivalence)

        add_test(
            NAME batched_mpi_multi_asset_smoke_local
            COMMAND batched_mpi_multi_asset_lob
                --duration-seconds 2
                --books 10
                --window-ms 1000
                --base-config ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
        )
        set_tests_properties(batched_mpi_multi_asset_smoke_local PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            TIMEOUT 120
        )

        add_test(
            NAME fragmented_mpi_smoke_local
            COMMAND fragmented_mpi_lob
                --duration-seconds 2
                --assets 8
                --window-ms 1000
                --base-config ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                --shock
                --shock-time-seconds 1
                --metrics-csv ${CMAKE_BINARY_DIR}/fragmented_mpi_smoke_metrics.csv
        )
        set_tests_properties(fragmented_mpi_smoke_local PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            TIMEOUT 120
        )

        add_test(
            NAME fragmented_local_mm_spread_response_local
            COMMAND fragmented_mpi_lob
                --duration-seconds 1
                --assets 4
                --window-ms 1000
                --base-config ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                --disable-shared-mm
                --disable-value-agent
                --local-mm-improvement-probability 0.25
                --local-mm-spread-elasticity 1.0
                --local-mm-max-improvement-probability 1.0
        )
        set_tests_properties(fragmented_local_mm_spread_response_local PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            TIMEOUT 120
        )

        add_test(
            NAME fragmented_local_mm_invalid_spread_elasticity_local
            COMMAND fragmented_mpi_lob
                --duration-seconds 1
                --assets 4
                --base-config ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                --local-mm-spread-elasticity -0.1
        )
        set_tests_properties(
            fragmented_local_mm_invalid_spread_elasticity_local PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            TIMEOUT 120
            WILL_FAIL TRUE
        )

        add_test(
            NAME fragmented_local_mm_invalid_probability_cap_local
            COMMAND fragmented_mpi_lob
                --duration-seconds 1
                --assets 4
                --base-config ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                --local-mm-improvement-probability 0.5
                --local-mm-max-improvement-probability 0.25
        )
        set_tests_properties(
            fragmented_local_mm_invalid_probability_cap_local PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            TIMEOUT 120
            WILL_FAIL TRUE
        )

        # The final financial case uses full empirical rows directly (not
        # replicated templates), one book per asset, relative shared quote sizes, and
        # a contemporaneous-bid-depth-normalised sell intervention.
        add_test(
            NAME fragmented_mpi_empirical_universe_local
            COMMAND fragmented_mpi_lob
                --duration-seconds 2
                --universe-config ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                --window-ms 1000
                --seed 20200130
                --shared-quote-relative
                --shared-quote-multiplier 1.0
                --disable-value-agent
                --shock
                --shock-time-seconds 1
                --shock-top-depth-multiple 1.0
                --metrics-csv ${CMAKE_BINARY_DIR}/fragmented_mpi_empirical_metrics.csv
                --shock-targets-csv ${CMAKE_BINARY_DIR}/fragmented_mpi_empirical_targets.csv
        )
        set_tests_properties(fragmented_mpi_empirical_universe_local PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            TIMEOUT 120
        )

        add_executable(test_fragmented_calibration_output
            tests/test_fragmented_calibration_output.cpp)
        target_link_libraries(test_fragmented_calibration_output PRIVATE dlob_core)
        add_test(NAME fragmented_calibration_output
                 COMMAND test_fragmented_calibration_output)

        foreach(policy_variant IN ITEMS legacy periodic news recheck gap)
            if(policy_variant STREQUAL "legacy")
                set(policy_file fragmented_value_policy.csv)
            else()
                set(policy_file fragmented_value_policy_${policy_variant}.csv)
            endif()
            add_test(
                NAME fragmented_value_policy_${policy_variant}_local
                COMMAND fragmented_mpi_lob
                    --duration-seconds 1
                    --universe-config ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                    --window-ms 1000
                    --disable-shared-mm
                    --disable-local-mm
                    --value-agent-policy-csv ${CMAKE_SOURCE_DIR}/tests/data/${policy_file}
            )
            set_tests_properties(
                fragmented_value_policy_${policy_variant}_local PROPERTIES
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                TIMEOUT 120
            )
        endforeach()
        add_test(
            NAME fragmented_value_policy_invalid_trigger_local
            COMMAND fragmented_mpi_lob
                --duration-seconds 1
                --universe-config ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                --window-ms 1000
                --disable-shared-mm
                --disable-local-mm
                --value-agent-policy-csv ${CMAKE_SOURCE_DIR}/tests/data/fragmented_value_policy_invalid_trigger.csv
        )
        set_tests_properties(
            fragmented_value_policy_invalid_trigger_local PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            TIMEOUT 120
            WILL_FAIL TRUE
        )
        add_test(
            NAME fragmented_value_policy_invalid_rechecks_local
            COMMAND fragmented_mpi_lob
                --duration-seconds 1
                --universe-config ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                --window-ms 1000
                --disable-shared-mm
                --disable-local-mm
                --value-agent-policy-csv ${CMAKE_SOURCE_DIR}/tests/data/fragmented_value_policy_invalid_rechecks.csv
        )
        set_tests_properties(
            fragmented_value_policy_invalid_rechecks_local PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            TIMEOUT 120
            WILL_FAIL TRUE
        )
        foreach(invalid_gap_variant IN ITEMS
                gap_elasticity gap_cap zero_gap_threshold)
            add_test(
                NAME fragmented_value_policy_invalid_${invalid_gap_variant}_local
                COMMAND fragmented_mpi_lob
                    --duration-seconds 1
                    --universe-config ${CMAKE_SOURCE_DIR}/config/qqq_aapl_msft_amzn_20200130.csv
                    --window-ms 1000
                    --disable-shared-mm
                    --disable-local-mm
                    --value-agent-policy-csv ${CMAKE_SOURCE_DIR}/tests/data/fragmented_value_policy_invalid_${invalid_gap_variant}.csv
            )
            set_tests_properties(
                fragmented_value_policy_invalid_${invalid_gap_variant}_local
                PROPERTIES
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                TIMEOUT 120
                WILL_FAIL TRUE
            )
        endforeach()

        add_executable(test_fragmented_model_semantics
            tests/test_fragmented_model_semantics.cpp)
        target_link_libraries(test_fragmented_model_semantics PRIVATE dlob_core)
        add_test(NAME fragmented_model_semantics
                 COMMAND test_fragmented_model_semantics)

        add_executable(test_liquidity_shock
            tests/test_liquidity_shock.cpp)
        target_link_libraries(test_liquidity_shock PRIVATE dlob_core)
        add_test(NAME liquidity_shock COMMAND test_liquidity_shock)

        add_executable(test_etf_arbitrage tests/test_etf_arbitrage.cpp)
        target_link_libraries(test_etf_arbitrage PRIVATE dlob_core)
        add_test(NAME etf_arbitrage COMMAND test_etf_arbitrage)

        add_executable(test_fundamental_value_agent
            tests/test_fundamental_value_agent.cpp)
        target_link_libraries(test_fundamental_value_agent PRIVATE dlob_core)
        add_test(NAME fundamental_value_agent COMMAND test_fundamental_value_agent)

        # These small correctness programs intentionally use the standard
        # assert macro.  Release normally defines NDEBUG, which would turn the
        # tests into empty smoke checks.  Keep assertions active in test
        # executables while leaving the measured simulator binaries unchanged.
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            set(LOB_ASSERTION_TEST_TARGETS
                test_event_ordering
                test_lob_reports
                test_population_partition
                test_calibration_parameters
                test_smc_math
                test_empirical_targets
                test_event_driven_population
                test_shared_market_snapshot
                test_trade_execution
                test_cancel_nearest_level
                test_background_quote_improvement
                test_background_hawkes_stream
                test_queue_reactive_background_policy
                test_trade_tape_hash
                test_shared_market_maker
                test_multiasset_ordering
                test_sequential_multiasset_determinism
                test_exact_mpi_equivalence
                test_fragmented_calibration_output
                test_fragmented_model_semantics
                test_liquidity_shock
                test_etf_arbitrage
                test_fundamental_value_agent
            )
            foreach(test_target IN LISTS LOB_ASSERTION_TEST_TARGETS)
                target_compile_options(${test_target} PRIVATE -UNDEBUG)
            endforeach()
        endif()

        find_package(Python3 QUIET COMPONENTS Interpreter)
        if(Python3_Interpreter_FOUND)
            add_test(
                NAME itch_extractor_python
                COMMAND ${Python3_EXECUTABLE} -m unittest -v
                        tests/test_itch_extractor.py
                        tests/test_assemble_itch50_universe_batches.py
                        tests/test_build_itch_universe_config.py
                        tests/test_derive_hawkes_rates.py
                        tests/test_compare_qqq_baseline.py
                        tests/test_build_multi_asset_config.py
                        tests/test_analyze_liquidity_shock.py
                        tests/test_validate_selected_exact_mpi.py
                        tests/test_value_agent_calibration_protocol.py
                        tests/test_summarize_value_validation.py
                        tests/test_cluster_empirical_universe.py
                        tests/test_cluster_value_agent_calibration.py
                        tests/test_pool_multiday_empirical_universe.py
                        tests/test_real_universe_case_study_submission_contract.py
                        tests/test_deterministic_build_contract.py
                        tests/test_run_local_multiday_calibration.py
                        tests/test_summarize_calibration_diagnostics.py
                        tests/test_analyze_fragmented_shared_liquidity_case.py
                        tests/test_analyze_cluster_liquidity_heterogeneity.py
                        tests/test_select_itch50_universe.py
                        tests/test_intersect_empirical_universe_configs.py
                        tests/test_run_fragmented_mpi_experiments.py
                        tests/test_certification_cohort.py
                        tests/test_empirical_calibration_preflight.py
                        tests/test_fixed_universe_workflow.py
                        tests/test_global_calibration_certification_verifier.py
                        tests/test_extract_itch50_symbols_queue_state.py
                        tests/test_prepare_queue_reactive_dated_configs.py
                        tests/test_fit_queue_reactive_hawkes_policy.py
                        tests/test_strict_model_validation.py
                        tests/test_queue_reactive_calibration_driver.py
                        tests/test_resolve_queue_reactive_case_artifact.py
            )
            set_tests_properties(itch_extractor_python PROPERTIES
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            )
        endif()

        add_test(
            NAME event_driven_smoke_local
            COMMAND distributed_lob
                --profile debug
                --duration-seconds 20
                --expected-ranks 1
                --event-driven
                --shared-snapshot
                --data-dir ${CMAKE_SOURCE_DIR}/data
                --output-dir ${CMAKE_BINARY_DIR}/test_event_driven_smoke_output
        )

        if(MPI_CXX_FOUND AND MPIEXEC_EXECUTABLE)
            add_test(
                NAME shared_market_snapshot_mpi_5
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 5
                        ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:test_shared_market_snapshot>
                        ${MPIEXEC_POSTFLAGS}
            )
            set_tests_properties(shared_market_snapshot_mpi_5 PROPERTIES
                PROCESSORS 5
                TIMEOUT 120
            )

            add_test(
                NAME event_driven_smoke_mpi_5
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 5
                        ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:distributed_lob>
                        ${MPIEXEC_POSTFLAGS}
                        --profile debug
                        --duration-seconds 10
                        --expected-ranks 5
                        --event-driven
                        --shared-snapshot
                        --max-wall-seconds 60
                        --data-dir ${CMAKE_SOURCE_DIR}/data
                        --output-dir ${CMAKE_BINARY_DIR}/test_event_driven_mpi_output
            )
            set_tests_properties(event_driven_smoke_mpi_5 PROPERTIES
                PROCESSORS 5
                TIMEOUT 120
            )

            foreach(rank_count IN ITEMS 2 3 4 5)
                add_test(
                    NAME exact_mpi_equivalence_mpi_${rank_count}
                    COMMAND ${MPIEXEC_EXECUTABLE}
                            ${MPIEXEC_NUMPROC_FLAG} ${rank_count}
                            ${MPIEXEC_PREFLAGS}
                            $<TARGET_FILE:test_exact_mpi_equivalence>
                            ${MPIEXEC_POSTFLAGS}
                )
                set_tests_properties(
                    exact_mpi_equivalence_mpi_${rank_count} PROPERTIES
                    PROCESSORS ${rank_count}
                    TIMEOUT 180
                )
            endforeach()
        endif()
    endif()
endif()
