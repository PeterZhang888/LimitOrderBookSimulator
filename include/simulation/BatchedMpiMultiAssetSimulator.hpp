// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#pragma once

#include "mpi/MpiCompat.hpp"
#include "simulation/MultiAssetTypes.hpp"

#include <cstdint>
#include <memory>

namespace dlob {

// Result for the deliberately coarse-grained benchmark model.  Local LOB
// events retain timestamp order, while the shared market maker and ETF
// arbitrageur observe all books only at fixed window boundaries.
struct BatchedMpiMultiAssetResult {
    int rank = 0;
    int world_size = 1;
    int book_count = 0;
    int local_book_count = 0;
    std::uint64_t windows = 0;
    std::uint64_t processed_orders = 0;
    std::uint64_t trades = 0;
    std::uint64_t market_maker_orders = 0;
    std::uint64_t arbitrage_orders = 0;
    std::uint64_t stale_snapshot_uses = 0;
    std::uint64_t state_hash = 0;
    double wall_seconds = 0.0;
    double initialization_seconds = 0.0;
    double max_compute_seconds = 0.0;
    double max_communication_seconds = 0.0;
    double controller_seconds = 0.0;
    double load_imbalance = 1.0;

    [[nodiscard]] bool is_root() const noexcept { return rank == 0; }
};

class BatchedMpiMultiAssetSimulator {
public:
    BatchedMpiMultiAssetSimulator(MPI_Comm communicator,
                                  SequentialMultiAssetConfig config,
                                  std::int64_t window_ns);
    ~BatchedMpiMultiAssetSimulator();

    BatchedMpiMultiAssetSimulator(const BatchedMpiMultiAssetSimulator&) = delete;
    BatchedMpiMultiAssetSimulator& operator=(const BatchedMpiMultiAssetSimulator&) = delete;

    [[nodiscard]] BatchedMpiMultiAssetResult run();
    [[nodiscard]] static int owner_rank(BookId book_id, int world_size);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dlob
