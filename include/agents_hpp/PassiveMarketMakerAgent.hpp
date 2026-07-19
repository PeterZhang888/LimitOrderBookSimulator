#pragma once

#include "common/AgentUtilities.hpp"

#include <cstdint>
#include <vector>

namespace dlob {

struct PassiveMarketMakerConfig {
    int tick_size = 100;
    int num_levels = 1;
    int order_quantity = 100;
    int level_spacing_ticks = 1;
    int min_spread_ticks = 2;
    std::int64_t quote_interval_ns = 20'000'000LL;
    double quote_skip_probability = 0.0;
    double recovery_quote_skip_probability = 0.0;
};

class PassiveMarketMakerAgent {
public:
    PassiveMarketMakerAgent(int owner_id,
                            const PassiveMarketMakerConfig& config,
                            std::int64_t simulation_start_ns,
                            std::uint64_t seed);

    void generate_orders(const MarketState& current,
                         std::int64_t window_start_ns,
                         std::int64_t window_end_ns,
                         OrderMessageBuilder& builder,
                         std::vector<OrderMessage>& out);

    void apply_report(const AgentReport& report);
    int owner_id() const { return owner_id_; }
    int inventory() const { return inventory_; }

private:
    int sample_quantity(Side side);
    int inventory_skew_ticks() const;

    int owner_id_ = 0;
    PassiveMarketMakerConfig config_{};
    std::int64_t next_wake_ns_ = 0;
    FastRng rng_;
    int inventory_ = 0;
    double cash_ticks_ = 0.0;
};

} // namespace dlob
