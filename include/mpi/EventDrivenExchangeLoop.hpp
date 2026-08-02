#pragma once

#include "agents/AgentPopulation.hpp"
#include "calibration/SimulationRecorder.hpp"
#include "common/DistributedTypes.hpp"
#include "exchange/BackgroundHawkesAgent.hpp"
#include "exchange/DistributedLimitOrderBook.hpp"
#include "mpi/MpiCompat.hpp"
#include "mpi/SharedMarketSnapshot.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <vector>

namespace dlob {

class PerformanceMetrics;

struct EventDrivenRunResult {
    MarketState closing_state{};
    std::size_t pending_orders = 0;
    std::size_t peak_pending_orders = 0;
    std::uint64_t activations = 0;
    bool terminated_early = false;
    std::int64_t final_time_ns = 0;
    std::string termination_reason;
};

class EventDrivenExchangeLoop {
public:
    EventDrivenExchangeLoop(MPI_Comm communicator,
                            int rank,
                            int world_size,
                            DistributedLimitOrderBook& book,
                            BackgroundHawkesAgent& background,
                            const std::vector<HawkesEvent>& hawkes_events,
                            AgentPopulation& population,
                            PerformanceMetrics& metrics,
                            calibration::SimulationRecorder* recorder,
                            SharedMarketSnapshot& shared_snapshot,
                            std::int64_t end_time_ns,
                            int tick_size,
                            std::uint64_t seed,
                            double max_wall_seconds);

    EventDrivenRunResult run();

private:
    struct OrderLater {
        bool operator()(const OrderMessage& left, const OrderMessage& right) const;
    };

    EventDrivenRunResult run_exchange();
    EventDrivenRunResult run_worker();
    EventDrivenRunResult run_single_process();

    void apply_market_event(const OrderMessage& message,
                            std::vector<std::vector<AgentReport>>& pending_reports);
    void accumulate_reports(std::vector<std::vector<AgentReport>>& pending_reports);
    void advance_fundamental(std::int64_t target_time_ns);

    MPI_Comm communicator_;
    int rank_ = 0;
    int world_size_ = 1;
    DistributedLimitOrderBook& book_;
    BackgroundHawkesAgent& background_;
    const std::vector<HawkesEvent>& hawkes_events_;
    AgentPopulation& population_;
    PerformanceMetrics& metrics_;
    calibration::SimulationRecorder* recorder_ = nullptr;
    SharedMarketSnapshot& shared_snapshot_;
    std::int64_t end_time_ns_ = 0;
    int tick_size_ = 1;

    std::size_t next_hawkes_ = 0;
    std::uint64_t background_sequence_ = 1;
    std::optional<OrderMessage> cached_background_;
    std::uint64_t activation_sequence_ = 1;
    std::priority_queue<OrderMessage, std::vector<OrderMessage>, OrderLater> pending_orders_;
    std::size_t peak_pending_orders_ = 0;

    std::mt19937_64 fundamental_rng_;
    std::normal_distribution<double> fundamental_shock_{0.0, 0.03};
    double fundamental_value_ = 2'203'550.0;
    std::int64_t fundamental_time_ns_ = 0;
    double max_wall_seconds_ = 0.0;
};

} // namespace dlob
