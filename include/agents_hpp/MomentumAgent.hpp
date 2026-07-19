#pragma once

#include "common/AgentUtilities.hpp"

#include <cstdint>
#include <vector>

namespace dlob {

struct MomentumAgentConfig {
    std::int64_t lookback_ns = 1'000'000'000LL;
    double wake_rate_per_second = 0.20;
    double threshold_ticks = 0.25;
    double order_flow_imbalance_threshold = 0.10;
    double depth_imbalance_threshold = 0.30;
    double strong_depth_imbalance_threshold = 0.50;
    int order_quantity = 100;
    int tick_size = 100;
};

class MomentumAgent {
public:
    MomentumAgent(int owner_id,
                  const MomentumAgentConfig& config,
                  std::int64_t simulation_start_ns,
                  std::uint64_t seed);

    void generate_orders(const MarketState& current,
                         const MarketState* past,
                         std::int64_t window_start_ns,
                         std::int64_t window_end_ns,
                         OrderMessageBuilder& builder,
                         std::vector<OrderMessage>& out);

    void apply_report(const AgentReport& report);
    int owner_id() const { return owner_id_; }

private:
    int owner_id_ = 0;
    MomentumAgentConfig config_{};
    std::int64_t next_wake_ns_ = 0;
    FastRng rng_;
    int inventory_ = 0;
    double cash_ticks_ = 0.0;
};

} // namespace dlob
