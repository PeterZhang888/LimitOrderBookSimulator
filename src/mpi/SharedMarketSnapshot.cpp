// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#include "mpi/SharedMarketSnapshot.hpp"

#include <stdexcept>
#include <string>

namespace dlob {
namespace {

#if LOB_HAS_REAL_MPI
void check_mpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed");
    }
}
#endif

} // namespace

SharedMarketSnapshot::SharedMarketSnapshot(MPI_Comm communicator,
                                           int rank,
                                           int world_size,
                                           bool requested)
    : rank_(rank) {
#if LOB_HAS_REAL_MPI
    if (!requested || world_size <= 1) return;

    check_mpi(MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &shared_comm_),
              "MPI_Comm_split_type(shared snapshot)");
    int shared_size = 0;
    check_mpi(MPI_Comm_size(shared_comm_, &shared_size),
              "MPI_Comm_size(shared snapshot)");
    if (shared_size != world_size) {
        MPI_Comm_free(&shared_comm_);
        shared_comm_ = MPI_COMM_NULL;
        return;
    }

    MPI_Aint bytes = rank == 0 ? static_cast<MPI_Aint>(sizeof(SharedMarketSnapshotSlot)) : 0;
    int displacement = 1;
    void* base = nullptr;
    const int allocation_status = MPI_Win_allocate_shared(
        bytes, displacement, MPI_INFO_NULL, shared_comm_, &base, &window_);
    if (allocation_status != MPI_SUCCESS) {
        MPI_Comm_free(&shared_comm_);
        shared_comm_ = MPI_COMM_NULL;
        check_mpi(allocation_status, "MPI_Win_allocate_shared");
    }

    if (rank == 0) {
        slot_ = static_cast<SharedMarketSnapshotSlot*>(base);
    } else {
        MPI_Aint queried_bytes = 0;
        int queried_disp = 0;
        void* queried = nullptr;
        if (MPI_Win_shared_query(window_, 0, &queried_bytes, &queried_disp,
                                 &queried) != MPI_SUCCESS
            || queried_bytes < static_cast<MPI_Aint>(sizeof(SharedMarketSnapshotSlot))) {
            MPI_Win_free(&window_);
            MPI_Comm_free(&shared_comm_);
            window_ = MPI_WIN_NULL;
            shared_comm_ = MPI_COMM_NULL;
            throw std::runtime_error("MPI_Win_shared_query failed");
        }
        slot_ = static_cast<SharedMarketSnapshotSlot*>(queried);
    }

    const int lock_status = MPI_Win_lock_all(0, window_);
    if (lock_status != MPI_SUCCESS) {
        MPI_Win_free(&window_);
        MPI_Comm_free(&shared_comm_);
        window_ = MPI_WIN_NULL;
        shared_comm_ = MPI_COMM_NULL;
        check_mpi(lock_status, "MPI_Win_lock_all");
    }
    if (rank == 0) *slot_ = SharedMarketSnapshotSlot{};
    check_mpi(MPI_Win_sync(window_), "MPI_Win_sync(initial snapshot)");
    check_mpi(MPI_Barrier(shared_comm_), "MPI_Barrier(initial snapshot)");
    enabled_ = true;
#else
    (void)communicator;
    enabled_ = requested && world_size > 1;
#endif
}

SharedMarketSnapshot::~SharedMarketSnapshot() {
#if LOB_HAS_REAL_MPI
    if (window_ != MPI_WIN_NULL) {
        MPI_Win_unlock_all(window_);
        MPI_Win_free(&window_);
    }
    if (shared_comm_ != MPI_COMM_NULL) MPI_Comm_free(&shared_comm_);
#endif
}

std::uint64_t SharedMarketSnapshot::publish(const MarketState& state) {
    if (rank_ != 0) throw std::logic_error("Only rank 0 may publish a shared market snapshot");
#if LOB_HAS_REAL_MPI
    if (enabled_) {
        const std::uint64_t next = slot_->version + 1;
        slot_->state = state;
        slot_->version = next;
        check_mpi(MPI_Win_sync(window_), "MPI_Win_sync(publish snapshot)");
        return next;
    }
#endif
    local_slot_.state = state;
    return ++local_slot_.version;
}

MarketState SharedMarketSnapshot::read(std::uint64_t expected_version) const {
#if LOB_HAS_REAL_MPI
    if (enabled_) {
        check_mpi(MPI_Win_sync(window_), "MPI_Win_sync(read snapshot)");
        if (slot_->version != expected_version) {
            throw std::runtime_error("Shared market snapshot version mismatch");
        }
        return slot_->state;
    }
#endif
    if (local_slot_.version != expected_version) {
        throw std::runtime_error("Local market snapshot version mismatch");
    }
    return local_slot_.state;
}

} // namespace dlob
