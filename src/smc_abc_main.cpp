// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#include "calibration/SmcAbc.hpp"
#include "mpi/MpiCompat.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed\n";
        return 1;
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        const dlob::calibration::SmcAbcConfig config =
            dlob::calibration::parse_smc_abc_config(argc, argv);
        dlob::calibration::run_smc_abc(MPI_COMM_WORLD, config);
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Rank " << rank << " SMC-ABC error: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }
}
