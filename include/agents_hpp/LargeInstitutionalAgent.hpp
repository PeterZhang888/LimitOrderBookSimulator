#pragma once

#include "common/AgentUtilities.hpp"

#include <cstdint>
#include <vector>

namespace dlob {

struct LargeInstitutionalConfig {
    Side side = Side::Buy;
    int parent_quantity = 50'000;
    int child_quantity = 500;
    double participation_cap = 0.10;
    double wake_rate_per_second = 0.01;
    std::int64_t start_time_ns = 0;
    std::int64_t end_time_ns = 60'000'000'000LL;
};

class LargeInstitutionalAgent {
public:
    LargeInstitutionalAgent(int owner_id,
                            const LargeInstitutionalConfig& config,
                            std::uint64_t seed);

    void generate_orders(const MarketState& current,
                         std::int64_t window_start_ns,
                         std::int64_t window_end_ns,
                         OrderMessageBuilder& builder,
                         std::vector<OrderMessage>& out);

    void apply_report(const AgentReport& report);
    int owner_id() const { return owner_id_; }
    int remaining_quantity() const;
    bool is_finished() const { return remaining_quantity() <= 0; }

private:
    int owner_id_ = 0;
    LargeInstitutionalConfig config_{};
    std::int64_t next_wake_ns_ = 0;
    FastRng rng_;
    int executed_quantity_ = 0;
    int outstanding_quantity_ = 0;
    int inventory_ = 0;
    double cash_ticks_ = 0.0;
};

} // namespace dlob
