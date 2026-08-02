// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#pragma once

#include "agents/InformedTraderAgent.hpp"
#include "agents/LargeInstitutionalAgent.hpp"
#include "agents/MomentumAgent.hpp"
#include "agents/PassiveMarketMakerAgent.hpp"

#include <cstdint>
#include <deque>
#include <queue>
#include <vector>

namespace dlob {

struct PopulationConfig {
    // Fixed population composition. These counts are scenario inputs rather than
    // calibration dimensions in the reduced eight-parameter model.
    int market_makers = 3;
    int momentum_traders = 6'000;
    int informed_traders = 2'900;
    int institutional_traders = 100;
    int population_scale = 1;

    int tick_size = 100;
    std::int64_t simulation_start_ns = 0;
    std::int64_t simulation_end_ns = 60'000'000'000LL;
    std::uint64_t seed = 12345;

    // Reduced calibration controls.
    int market_maker_order_quantity = 100;
    int market_maker_min_spread_ticks = 2;
    std::int64_t market_maker_interval_ns = 20'000'000LL;
    double market_maker_quote_skip_probability = 0.05;

    double momentum_rate_per_second = 0.20;
    double momentum_threshold_ticks = 0.25;
    int momentum_order_quantity = 100;

    double informed_rate_per_second = 0.05;
    double informed_signal_noise_ticks = 1.5;
    double informed_trade_threshold_ticks = 1.0;
    int informed_base_quantity = 100;

    double institutional_rate_per_second = 0.01;
    double institutional_participation_cap = 0.10;

    // Rank-level micro-batching lookahead. Orders retain their exact generated
    // and arrival timestamps; only the market snapshot is held constant within
    // one local activation horizon.
    std::int64_t market_maker_batch_horizon_ns = 100'000'000LL;
    std::int64_t momentum_batch_horizon_ns = 250'000'000LL;
    std::int64_t informed_batch_horizon_ns = 250'000'000LL;
    std::int64_t institutional_batch_horizon_ns = 1'000'000'000LL;
};

struct PopulationSummary {
    int market_makers = 0;
    int momentum = 0;
    int informed = 0;
    int institutional = 0;
    int total() const { return market_makers + momentum + informed + institutional; }
};

class AgentPopulation {
public:
    AgentPopulation(int mpi_rank, int world_size, const PopulationConfig& config);

    void observe_market(const MarketState& state);

    // Legacy fixed-window interface retained for benchmark comparison.
    std::vector<OrderMessage> generate_orders(std::int64_t window_start_ns,
                                              std::int64_t window_end_ns);

    // Event-driven interface. Processes only local agents whose scheduled wake
    // time is at or before cutoff_ns and returns one timestamped rank-level batch.
    std::vector<OrderMessage> generate_due_orders(std::int64_t activation_time_ns,
                                                  std::int64_t cutoff_ns);

    void apply_reports(const std::vector<AgentReport>& reports);

    std::int64_t next_wake_time() const;
    std::int64_t batch_horizon_ns() const;
    WorkerRole role() const noexcept { return role_; }

    PopulationSummary local_summary() const { return local_summary_; }
    PopulationSummary global_summary() const { return global_summary_; }
    bool is_worker() const { return is_worker_; }

private:
    struct AgentRef {
        AgentKind kind = AgentKind::Background;
        int index = -1;
    };

    struct WakeEntry {
        std::int64_t time_ns = no_wake_time;
        AgentKind kind = AgentKind::Background;
        int index = -1;
    };

    struct WakeLater {
        bool operator()(const WakeEntry& left, const WakeEntry& right) const {
            if (left.time_ns != right.time_ns) return left.time_ns > right.time_ns;
            if (left.kind != right.kind) {
                return static_cast<std::int32_t>(left.kind)
                    > static_cast<std::int32_t>(right.kind);
            }
            return left.index > right.index;
        }
    };

    static int local_count(int global_count, int worker_index, int worker_count);
    static WorkerRole role_for_rank(int rank, int world_size);
    static AgentKind role_agent_kind(WorkerRole role);

    int role_worker_index(WorkerRole role) const;
    int role_worker_count(WorkerRole role) const;
    bool owns_kind(AgentKind kind) const;

    const MarketState* past_state(std::int64_t target_ns) const;
    void register_agent(int owner_id, AgentKind kind, int vector_index);
    void rebuild_wake_queue();
    void push_wake(AgentKind kind, int index);
    std::int64_t agent_next_wake(AgentKind kind, int index) const;
    void generate_one_due(const WakeEntry& entry,
                          std::int64_t activation_time_ns,
                          std::int64_t cutoff_ns,
                          std::vector<OrderMessage>& output);

    int rank_ = 0;
    int world_size_ = 1;
    bool is_worker_ = false;
    WorkerRole role_ = WorkerRole::Exchange;
    PopulationConfig config_{};
    PopulationSummary local_summary_{};
    PopulationSummary global_summary_{};
    OrderMessageBuilder message_builder_;
    MarketState current_state_{};
    std::deque<MarketState> history_;

    std::vector<MomentumAgent> momentum_;
    std::vector<InformedTraderAgent> informed_;
    std::vector<LargeInstitutionalAgent> institutional_;
    std::vector<PassiveMarketMakerAgent> market_makers_;
    std::vector<AgentRef> owner_table_;
    std::priority_queue<WakeEntry, std::vector<WakeEntry>, WakeLater> wake_queue_;
};

} // namespace dlob
