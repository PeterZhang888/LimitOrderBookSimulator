#include "mpi/MpiTransport.hpp"

#include "mpi/MpiCompat.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace dlob {
namespace {

int checked_bytes(std::size_t count, std::size_t item_size, const char* label) {
    if (item_size > 0 && count > static_cast<std::size_t>(std::numeric_limits<int>::max()) / item_size) {
        throw std::runtime_error(std::string(label) + " exceeds MPI int byte-count limit");
    }
    return static_cast<int>(count * item_size);
}

void check_mpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) throw std::runtime_error(std::string(operation) + " failed");
}

} // namespace

std::vector<OrderMessage> gather_orders(const std::vector<OrderMessage>& local_orders,
                                        int rank,
                                        int world_size,
                                        PerformanceMetrics& metrics) {
    static_assert(std::is_trivially_copyable_v<OrderMessage>);
    const int local_bytes = checked_bytes(local_orders.size(), sizeof(OrderMessage), "local order batch");
    metrics.counters().order_bytes_sent += static_cast<std::uint64_t>(local_bytes);

    std::vector<int> receive_counts;
    if (rank == 0) receive_counts.resize(static_cast<std::size_t>(world_size), 0);

    {
        ScopedStageTimer timer(metrics, TimingStage::GatherOrderCounts);
        check_mpi(MPI_Gather(&local_bytes, 1, MPI_INT,
                             rank == 0 ? receive_counts.data() : nullptr,
                             1, MPI_INT, 0, MPI_COMM_WORLD),
                  "MPI_Gather(order counts)");
    }

    std::vector<int> displacements;
    std::vector<OrderMessage> gathered;
    if (rank == 0) {
        displacements.resize(static_cast<std::size_t>(world_size), 0);
        std::int64_t total_bytes = 0;
        for (int r = 0; r < world_size; ++r) {
            const int count = receive_counts[static_cast<std::size_t>(r)];
            if (count < 0 || count % static_cast<int>(sizeof(OrderMessage)) != 0) {
                throw std::runtime_error("Invalid order byte count received from MPI rank " + std::to_string(r));
            }
            if (total_bytes > std::numeric_limits<int>::max() - count) {
                throw std::runtime_error("Gathered order payload exceeds MPI_Gatherv int displacement limit");
            }
            displacements[static_cast<std::size_t>(r)] = static_cast<int>(total_bytes);
            total_bytes += count;
        }
        if (total_bytes % static_cast<std::int64_t>(sizeof(OrderMessage)) != 0) {
            throw std::runtime_error("Gathered order payload is not aligned to OrderMessage size");
        }
        gathered.resize(static_cast<std::size_t>(total_bytes / static_cast<std::int64_t>(sizeof(OrderMessage))));
        metrics.counters().order_bytes_received_exchange += static_cast<std::uint64_t>(total_bytes);
        metrics.counters().strategic_orders_received_exchange += gathered.size();
    }

    {
        ScopedStageTimer timer(metrics, TimingStage::GatherOrderPayload);
        check_mpi(MPI_Gatherv(local_orders.empty() ? nullptr : local_orders.data(),
                              local_bytes, MPI_BYTE,
                              rank == 0 && !gathered.empty() ? gathered.data() : nullptr,
                              rank == 0 ? receive_counts.data() : nullptr,
                              rank == 0 ? displacements.data() : nullptr,
                              MPI_BYTE, 0, MPI_COMM_WORLD),
                  "MPI_Gatherv(order payload)");
    }
    return gathered;
}

std::vector<AgentReport> scatter_reports(const std::vector<AgentReport>& root_reports,
                                         int rank,
                                         int world_size,
                                         PerformanceMetrics& metrics) {
    static_assert(std::is_trivially_copyable_v<AgentReport>);
    std::vector<int> send_counts_bytes;
    std::vector<int> displacements_bytes;
    std::vector<AgentReport> packed;

    if (rank == 0) {
        ScopedStageTimer timer(metrics, TimingStage::ReportPacking);
        std::vector<std::vector<AgentReport>> by_rank(static_cast<std::size_t>(world_size));
        for (const AgentReport& report : root_reports) {
            const int destination = owner_rank(report.owner_id);
            if (destination > 0 && destination < world_size) {
                by_rank[static_cast<std::size_t>(destination)].push_back(report);
            } else if (world_size == 1 && destination == 0) {
                by_rank[0].push_back(report);
            }
        }

        send_counts_bytes.resize(static_cast<std::size_t>(world_size), 0);
        displacements_bytes.resize(static_cast<std::size_t>(world_size), 0);
        std::int64_t total_items = 0;
        for (int r = 0; r < world_size; ++r) {
            const int bytes = checked_bytes(by_rank[static_cast<std::size_t>(r)].size(),
                                            sizeof(AgentReport), "report batch");
            const std::int64_t displacement = total_items * static_cast<std::int64_t>(sizeof(AgentReport));
            if (displacement > std::numeric_limits<int>::max()) {
                throw std::runtime_error("Scattered report payload exceeds MPI_Scatterv int displacement limit");
            }
            displacements_bytes[static_cast<std::size_t>(r)] = static_cast<int>(displacement);
            send_counts_bytes[static_cast<std::size_t>(r)] = bytes;
            total_items += static_cast<std::int64_t>(by_rank[static_cast<std::size_t>(r)].size());
        }
        packed.reserve(static_cast<std::size_t>(total_items));
        for (int r = 0; r < world_size; ++r) {
            const auto& bucket = by_rank[static_cast<std::size_t>(r)];
            packed.insert(packed.end(), bucket.begin(), bucket.end());
        }
        metrics.counters().report_bytes_sent_exchange +=
            static_cast<std::uint64_t>(packed.size() * sizeof(AgentReport));
    }

    int receive_bytes = 0;
    {
        ScopedStageTimer timer(metrics, TimingStage::ScatterReportCounts);
        check_mpi(MPI_Scatter(rank == 0 ? send_counts_bytes.data() : nullptr,
                              1, MPI_INT, &receive_bytes, 1, MPI_INT, 0, MPI_COMM_WORLD),
                  "MPI_Scatter(report counts)");
    }
    if (receive_bytes < 0 || receive_bytes % static_cast<int>(sizeof(AgentReport)) != 0) {
        throw std::runtime_error("Invalid report byte count received by rank " + std::to_string(rank));
    }

    std::vector<AgentReport> received(static_cast<std::size_t>(receive_bytes) / sizeof(AgentReport));
    {
        ScopedStageTimer timer(metrics, TimingStage::ScatterReportPayload);
        check_mpi(MPI_Scatterv(rank == 0 && !packed.empty() ? packed.data() : nullptr,
                               rank == 0 ? send_counts_bytes.data() : nullptr,
                               rank == 0 ? displacements_bytes.data() : nullptr,
                               MPI_BYTE,
                               received.empty() ? nullptr : received.data(),
                               receive_bytes,
                               MPI_BYTE,
                               0,
                               MPI_COMM_WORLD),
                  "MPI_Scatterv(report payload)");
    }

    metrics.counters().report_bytes_received += static_cast<std::uint64_t>(receive_bytes);
    metrics.counters().reports_received += received.size();
    return received;
}

} // namespace dlob
