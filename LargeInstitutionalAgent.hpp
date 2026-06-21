#ifndef LARGE_INSTITUTIONAL_AGENT_HPP
#define LARGE_INSTITUTIONAL_AGENT_HPP

#include "LimitOrderBook.hpp"
#include "Order.hpp"

#include <cstdint>

class LargeInstitutionalAgent {
private:
    static int number_of_agents_;

    int agent_index_;

    Side side_;
    int parent_quantity_;
    int remaining_quantity_;
    int child_quantity_;
    std::int64_t start_time_ns_;
    std::int64_t end_time_ns_;
    bool active_;

public:
    LargeInstitutionalAgent(
        Side side,
        int parent_quantity,
        int child_quantity,
        std::int64_t start_time_ns,
        std::int64_t end_time_ns
    );

    LargeInstitutionalAgent(const LargeInstitutionalAgent&) = delete;
    LargeInstitutionalAgent& operator=(const LargeInstitutionalAgent&) = delete;

    LargeInstitutionalAgent(LargeInstitutionalAgent&&) = default;
    LargeInstitutionalAgent& operator=(LargeInstitutionalAgent&&) = default;

    void wake_up(
        LimitOrderBook& book,
        std::int64_t current_time_ns
    );

    bool is_finished() const;

    int remaining_quantity() const;

    int agent_index() const;

    static int number_of_agents();
};

#endif
