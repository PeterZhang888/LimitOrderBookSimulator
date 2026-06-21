#include "LargeInstitutionalAgent.hpp"

#include <algorithm>

int LargeInstitutionalAgent::number_of_agents_ = 0;

LargeInstitutionalAgent::LargeInstitutionalAgent(
    Side side,
    int parent_quantity,
    int child_quantity,
    std::int64_t start_time_ns,
    std::int64_t end_time_ns
)
    : agent_index_(number_of_agents_++),
      side_(side),
      parent_quantity_(parent_quantity),
      remaining_quantity_(parent_quantity),
      child_quantity_(child_quantity),
      start_time_ns_(start_time_ns),
      end_time_ns_(end_time_ns),
      active_(true)
{
}

void LargeInstitutionalAgent::wake_up(
    LimitOrderBook& book,
    std::int64_t current_time_ns
) {
    if (!active_) {
        return;
    }

    if (current_time_ns < start_time_ns_ || current_time_ns > end_time_ns_) {
        return;
    }

    if (remaining_quantity_ <= 0) {
        active_ = false;
        return;
    }

    int quantity_to_execute =
        std::min(child_quantity_, remaining_quantity_);

    int executed = 0;

    if (side_ == Side::Buy) {
        if (book.has_ask()) {
            executed = book.submit_market_order(
                Side::Buy,
                quantity_to_execute
            );
        }
    } else {
        if (book.has_bid()) {
            executed = book.submit_market_order(
                Side::Sell,
                quantity_to_execute
            );
        }
    }

    remaining_quantity_ -= executed;

    if (remaining_quantity_ <= 0) {
        active_ = false;
    }
}

bool LargeInstitutionalAgent::is_finished() const {
    return !active_ || remaining_quantity_ <= 0;
}

int LargeInstitutionalAgent::remaining_quantity() const {
    return remaining_quantity_;
}

int LargeInstitutionalAgent::agent_index() const {
    return agent_index_;
}

int LargeInstitutionalAgent::number_of_agents() {
    return number_of_agents_;
}
