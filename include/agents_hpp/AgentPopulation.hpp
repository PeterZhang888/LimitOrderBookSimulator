#pragma once

#include "agents/InformedTraderAgent.hpp"
#include "agents/LargeInstitutionalAgent.hpp"
#include "agents/MomentumAgent.hpp"
#include "agents/PassiveMarketMakerAgent.hpp"

#include <cstdint>
#include <deque>
#include <vector>

namespace dlob {

struct PopulationConfig {
    // Literature prior: 6,000 chartist traders, 3,000 value-oriented traders,
    // and 3 market makers. One hundred value-oriented agents are represented
    // explicitly as large institutional execution agents.
    int market_makers = 3;
    int momentum_traders = 6'000;
    int informed_traders = 2'900;
    int institutional_traders = 100;
    int population_scale = 1;

    int tick_size = 100;
    std::int64_t simulation_start_ns = 0;
    std::int64_t simulation_end_ns = 60'000'000'000LL;
    std::uint64_t seed = 12345;

    double momentum_rate_per_second = 0.20;
    double informed_rate_per_second = 0.05;
    double institutional_rate_per_second = 0.01;
    std::int64_t market_maker_interval_ns = 20'000'000LL;
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
    std::vector<OrderMessage> generate_orders(std::int64_t window_start_ns,
                                              std::int64_t window_end_ns);
    void apply_reports(const std::vector<AgentReport>& reports);

    PopulationSummary local_summary() const { return local_summary_; }
    PopulationSummary global_summary() const { return global_summary_; }
    bool is_worker() const { return is_worker_; }

private:
    struct AgentRef {
        AgentKind kind = AgentKind::Background;
        int index = -1;
    };

    static int local_count(int global_count, int worker_index, int worker_count);
    const MarketState* past_state(std::int64_t target_ns) const;
    void register_agent(int owner_id, AgentKind kind, int vector_index);

    int rank_ = 0;
    int world_size_ = 1;
    int worker_index_ = -1;
    int worker_count_ = 0;
    bool is_worker_ = false;
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
};

} // namespace dlob
