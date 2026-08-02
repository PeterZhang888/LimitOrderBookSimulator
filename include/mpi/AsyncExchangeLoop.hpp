// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#pragma once

#include "common/DistributedTypes.hpp"
#include "exchange/BackgroundHawkesAgent.hpp"
#include "exchange/DistributedLimitOrderBook.hpp"
#include "mpi/MpiCompat.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace dlob {

class PerformanceMetrics;
namespace calibration { class SimulationRecorder; }

struct ExchangeWindowResult {
    MarketState closing_state{};
    std::vector<AgentReport> local_reports;
};

class AsyncExchangeLoop {
public:
    AsyncExchangeLoop(
        MPI_Comm communicator,
        int rank,
        int world_size,
        DistributedLimitOrderBook& book,
        BackgroundHawkesAgent& background,
        const std::vector<HawkesEvent>& hawkes_events,
        PerformanceMetrics& metrics,
        calibration::SimulationRecorder* recorder = nullptr
    );

    ExchangeWindowResult run_exchange_window(
        std::uint64_t window_index,
        const MarketState& opening_state,
        std::int64_t window_end_ns,
        double fundamental_value,
        const std::vector<OrderMessage>& local_orders = {}
    );

    MarketState receive_market_state(std::uint64_t expected_window_index);
    void post_order_batch(std::uint64_t window_index,
                          const std::vector<OrderMessage>& orders);
    std::vector<AgentReport> receive_report_batch(
        std::uint64_t expected_window_index);

    std::size_t pending_order_count() const noexcept;
    std::size_t peak_pending_order_count() const noexcept;

private:
    std::vector<OrderMessage> receive_order_batches(
        std::uint64_t expected_window_index,
        const std::vector<OrderMessage>& local_orders,
        std::vector<MPI_Request>& state_send_requests
    );

    ExchangeWindowResult process_exchange_orders(
        std::int64_t window_end_ns,
        double fundamental_value,
        std::vector<OrderMessage> received_orders
    );

    void send_report_batches(
        std::uint64_t window_index,
        const std::vector<AgentReport>& reports,
        std::vector<AgentReport>& local_reports
    );

    MPI_Comm communicator_;
    int rank_;
    int world_size_;
    DistributedLimitOrderBook& book_;
    BackgroundHawkesAgent& background_;
    const std::vector<HawkesEvent>& hawkes_events_;
    PerformanceMetrics& metrics_;
    calibration::SimulationRecorder* recorder_ = nullptr;

    std::size_t next_hawkes_ = 0;
    std::optional<OrderMessage> cached_background_message_;
    std::uint64_t background_sequence_ = 1;
    std::vector<OrderMessage> pending_orders_;
    std::size_t peak_pending_orders_ = 0;
    std::vector<unsigned char> worker_order_send_buffer_;
    MPI_Request worker_order_send_request_ = MPI_REQUEST_NULL;
    bool worker_order_send_active_ = false;
};

} // namespace dlob
