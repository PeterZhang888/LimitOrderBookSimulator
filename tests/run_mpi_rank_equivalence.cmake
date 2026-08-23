if(NOT MPIEXEC_EXECUTABLE OR NOT MPIEXEC_NUMPROC_FLAG
   OR NOT TEST_EXECUTABLE OR NOT TEST_ROOT)
    message(FATAL_ERROR "missing MPI rank-equivalence test argument")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")

execute_process(
    COMMAND "${MPIEXEC_EXECUTABLE}" --version
    OUTPUT_VARIABLE MPIEXEC_VERSION
    ERROR_VARIABLE MPIEXEC_VERSION_ERROR)
set(OPENMPI_EXECUTION FALSE)
if(MPIEXEC_VERSION MATCHES "Open MPI"
   OR MPIEXEC_VERSION_ERROR MATCHES "Open MPI")
    set(OPENMPI_EXECUTION TRUE)
endif()

foreach(RANKS 1 2 4)
    set(RUN_ROOT "${TEST_ROOT}/ranks_${RANKS}")
    set(MPIEXEC_EXTRA_ARGS)
    if(OPENMPI_EXECUTION)
        list(APPEND MPIEXEC_EXTRA_ARGS
            --oversubscribe --map-by "ppr:${RANKS}:node" --bind-to none)
    endif()
    execute_process(
        COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_EXTRA_ARGS}
                "${MPIEXEC_NUMPROC_FLAG}" "${RANKS}"
                "${TEST_EXECUTABLE}" "${RUN_ROOT}"
        RESULT_VARIABLE STATUS
        OUTPUT_VARIABLE STANDARD_OUTPUT
        ERROR_VARIABLE STANDARD_ERROR)
    if(NOT STATUS EQUAL 0)
        message(FATAL_ERROR
            "${RANKS}-rank integration run failed (${STATUS})\n"
            "stdout:\n${STANDARD_OUTPUT}\n"
            "stderr:\n${STANDARD_ERROR}")
    endif()
endforeach()

foreach(CANDIDATE 2 4)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E compare_files
                "${TEST_ROOT}/ranks_1/canonical.txt"
                "${TEST_ROOT}/ranks_${CANDIDATE}/canonical.txt"
        RESULT_VARIABLE CANONICAL_STATUS)
    if(NOT CANONICAL_STATUS EQUAL 0)
        message(FATAL_ERROR
            "canonical simulator result differs between 1 and ${CANDIDATE} ranks")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E compare_files
                "${TEST_ROOT}/ranks_1/assets.csv"
                "${TEST_ROOT}/ranks_${CANDIDATE}/assets.csv"
        RESULT_VARIABLE ASSET_STATUS)
    if(NOT ASSET_STATUS EQUAL 0)
        message(FATAL_ERROR
            "per-asset simulator result differs between 1 and ${CANDIDATE} ranks")
    endif()
endforeach()
