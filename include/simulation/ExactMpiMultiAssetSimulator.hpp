#pragma once

#include "mpi/MpiCompat.hpp"
#include "simulation/MultiAssetTypes.hpp"

#include <memory>

namespace dlob {

// Result returned on every rank.  `model` is the same rank-independent model
// summary produced by the sequential reference; only wall time is expected to
// differ.  Rank zero additionally writes model.summary_csv.
struct ExactMpiMultiAssetResult {
    SequentialMultiAssetResult model;
    int rank = 0;
    int world_size = 1;
    int local_book_count = 0;

    [[nodiscard]] bool is_root() const noexcept { return rank == 0; }
};

// Conservative exact parallel discrete-event oracle for the interacting
// multi-asset model.  A book is owned by BookId % world_size.  Ranks retain
// only their local LOB/background/recorder state, while the shared market
// maker is replicated and advanced by the same globally broadcast events.
class ExactMpiMultiAssetSimulator {
public:
    ExactMpiMultiAssetSimulator(MPI_Comm communicator,
                                SequentialMultiAssetConfig config);
    ~ExactMpiMultiAssetSimulator();

    ExactMpiMultiAssetSimulator(const ExactMpiMultiAssetSimulator&) = delete;
    ExactMpiMultiAssetSimulator& operator=(const ExactMpiMultiAssetSimulator&) = delete;
    ExactMpiMultiAssetSimulator(ExactMpiMultiAssetSimulator&&) = delete;
    ExactMpiMultiAssetSimulator& operator=(ExactMpiMultiAssetSimulator&&) = delete;

    [[nodiscard]] ExactMpiMultiAssetResult run();

    [[nodiscard]] static int owner_rank(BookId book_id, int world_size);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dlob
