#pragma once

#include "LimitOrderBook.hpp"

#include <cstdint>
#include <random>

class LargeInstitutionalAgent {
public:
    LargeInstitutionalAgent(
        Side side,
        int parent_quantity,
        int child_quantity,
        double participation_cap,
        std::int64_t start_time_ns,
        std::int64_t end_time_ns,
        std::uint64_t seed
    );

    int wake_up(LimitOrderBook& book, std::int64_t current_time_ns);
    bool is_finished() const;
    int remaining_quantity() const;

private:
    int agent_index_ = 0;
    Side side_ = Side::Buy;
    int parent_quantity_ = 0;
    int remaining_quantity_ = 0;
    int child_quantity_ = 0;
    double participation_cap_ = 0.2;
    std::int64_t start_time_ns_ = 0;
    std::int64_t end_time_ns_ = 0;
    bool active_ = true;
    std::mt19937_64 rng_;

    static int number_of_agents_;
};
