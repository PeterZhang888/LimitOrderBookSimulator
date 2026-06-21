#ifndef MOMENTUM_AGENT_HPP
#define MOMENTUM_AGENT_HPP

#include "LimitOrderBook.hpp"

#include <cstdint>
#include <deque>

class MomentumAgent {
private:
    static int number_of_agents_;

    int agent_index_;

    struct MidRecord {
        std::int64_t time_ns;
        double mid_price;
    };

    std::deque<MidRecord> mid_history_;

    std::int64_t lookback_ns_;
    int order_quantity_;
    double threshold_ticks_;
    int tick_size_;

public:
    MomentumAgent(
        std::int64_t lookback_ns,
        int order_quantity,
        double threshold_ticks,
        int tick_size
    );

    MomentumAgent(const MomentumAgent&) = delete;
    MomentumAgent& operator=(const MomentumAgent&) = delete;

    MomentumAgent(MomentumAgent&&)= default;
    MomentumAgent& operator=(MomentumAgent&&)= default;

    void record_mid_price(
        const LimitOrderBook& book,
        std::int64_t current_time_ns
    );

    void wake_up(
        LimitOrderBook& book,
        std::int64_t current_time_ns
    );

    int agent_index() const;

    static int number_of_agents();
};

#endif
