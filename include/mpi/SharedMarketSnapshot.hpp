#pragma once

#include "common/DistributedTypes.hpp"
#include "mpi/MpiCompat.hpp"

#include <cstdint>

namespace dlob {

// Optional node-local shared snapshot. It is enabled only when every rank in
// the simulator communicator resides on the same physical node. The full LOB
// remains private to rank 0; workers can read only this compact immutable view.
class SharedMarketSnapshot {
public:
    SharedMarketSnapshot(MPI_Comm communicator, int rank, int world_size, bool requested);
    ~SharedMarketSnapshot();

    SharedMarketSnapshot(const SharedMarketSnapshot&) = delete;
    SharedMarketSnapshot& operator=(const SharedMarketSnapshot&) = delete;

    bool enabled() const noexcept { return enabled_; }
    std::uint64_t publish(const MarketState& state);
    MarketState read(std::uint64_t expected_version) const;

private:
    int rank_ = 0;
    bool enabled_ = false;
    mutable SharedMarketSnapshotSlot local_slot_{};

#if LOB_HAS_REAL_MPI
    MPI_Comm shared_comm_ = MPI_COMM_NULL;
    MPI_Win window_ = MPI_WIN_NULL;
    SharedMarketSnapshotSlot* slot_ = nullptr;
#endif
};

} // namespace dlob
