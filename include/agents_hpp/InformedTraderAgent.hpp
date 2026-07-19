#pragma once

#include "common/AgentUtilities.hpp"

#include <cstdint>
#include <vector>

namespace dlob {

struct InformedTraderConfig {
    double wake_rate_per_second = 0.05;
    double signal_noise_ticks = 1.5;
    double trade_threshold_ticks = 1.0;
    int base_quantity = 100;
    int max_abs_inventory = 5'000;
    int tick_size = 100;
};

class InformedTraderAgent {
public:
    InformedTraderAgent(int owner_id,
                        const InformedTraderConfig& config,
                        std::int64_t simulation_start_ns,
                        std::uint64_t seed);

    void generate_orders(const MarketState& current,
                         std::int64_t window_start_ns,
                         std::int64_t window_end_ns,
                         OrderMessageBuilder& builder,
                         std::vector<OrderMessage>& out);

    void apply_report(const AgentReport& report);
    int owner_id() const { return owner_id_; }

private:
    int owner_id_ = 0;
    InformedTraderConfig config_{};
    std::int64_t next_wake_ns_ = 0;
    FastRng rng_;
    int inventory_ = 0;
    double cash_ticks_ = 0.0;
};

} // namespace dlob
