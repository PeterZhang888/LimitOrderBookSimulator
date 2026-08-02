#include "mpi/SharedMarketSnapshot.hpp"

#include <cassert>
#include <cstdint>

int main(int argc, char** argv) {
    assert(MPI_Init(&argc, &argv) == MPI_SUCCESS);

    int rank = 0;
    int world_size = 1;
    assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
    assert(MPI_Comm_size(MPI_COMM_WORLD, &world_size) == MPI_SUCCESS);

    {
        dlob::SharedMarketSnapshot snapshot(
            MPI_COMM_WORLD, rank, world_size, true);

        dlob::MarketState expected;
        expected.exchange_time_ns = 123456789;
        expected.best_bid_ticks = 2203000;
        expected.best_ask_ticks = 2203100;
        expected.best_bid_depth = 500;
        expected.best_ask_depth = 700;
        expected.background_best_bid_depth = 450;
        expected.background_best_ask_depth = 650;
        expected.total_background_bid_depth = 4'500;
        expected.total_background_ask_depth = 6'500;
        expected.mid_price_ticks = 2203050.0;

        std::uint64_t version = 0;
        if (rank == 0) version = snapshot.publish(expected);
        assert(MPI_Bcast(&version, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD) == MPI_SUCCESS);
        assert(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);

        const dlob::MarketState observed = snapshot.read(version);
        assert(version == 1);
        assert(observed.exchange_time_ns == expected.exchange_time_ns);
        assert(observed.best_bid_ticks == expected.best_bid_ticks);
        assert(observed.best_ask_ticks == expected.best_ask_ticks);
        assert(observed.best_bid_depth == expected.best_bid_depth);
        assert(observed.best_ask_depth == expected.best_ask_depth);
        assert(observed.background_best_bid_depth
               == expected.background_best_bid_depth);
        assert(observed.background_best_ask_depth
               == expected.background_best_ask_depth);
        assert(observed.total_background_bid_depth
               == expected.total_background_bid_depth);
        assert(observed.total_background_ask_depth
               == expected.total_background_ask_depth);
        assert(observed.mid_price_ticks == expected.mid_price_ticks);

        assert(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
    }

    assert(MPI_Finalize() == MPI_SUCCESS);
    return 0;
}
