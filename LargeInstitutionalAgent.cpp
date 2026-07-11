#include "LargeInstitutionalAgent.hpp"

#include "EmpiricalDistribution.hpp"

#include <algorithm>
#include <cmath>

int LargeInstitutionalAgent::number_of_agents_ = 0;

namespace {
struct InstitutionalDistributions {
    EmpiricalDistribution market_buy_quantity;
    EmpiricalDistribution market_sell_quantity;
    InstitutionalDistributions() {
        market_buy_quantity.set_fallback(50, 1000);
        market_sell_quantity.set_fallback(50, 1000);
        market_buy_quantity.load_from_csv("market_buy_quantity_distribution.txt", "quantity");
        market_sell_quantity.load_from_csv("market_sell_quantity_distribution.txt", "quantity");
    }
};

InstitutionalDistributions& institutional_distributions() {
    static InstitutionalDistributions d;
    return d;
}

int sample_child_quantity(Side side, std::mt19937_64& rng) {
    auto& d = institutional_distributions();
    return side == Side::Buy ? d.market_buy_quantity.sample(rng) : d.market_sell_quantity.sample(rng);
}

int best_level_participation_cap(const LimitOrderBook& book, Side side, double cap) {
    int available = 0;
    if (side == Side::Buy) {
        if (!book.has_ask()) return 0;
        available = book.quantity_at_best_ask();
    } else {
        if (!book.has_bid()) return 0;
        available = book.quantity_at_best_bid();
    }
    if (available <= 0) return 0;
    return std::max(1, static_cast<int>(std::floor(std::max(0.01, cap) * static_cast<double>(available))));
}
}

LargeInstitutionalAgent::LargeInstitutionalAgent(
    Side side,
    int parent_quantity,
    int child_quantity,
    double participation_cap,
    std::int64_t start_time_ns,
    std::int64_t end_time_ns,
    std::uint64_t seed
)
    : agent_index_(number_of_agents_++),
      side_(side),
      parent_quantity_(std::max(1, parent_quantity)),
      remaining_quantity_(std::max(1, parent_quantity)),
      child_quantity_(std::max(1, child_quantity)),
      participation_cap_(std::max(0.01, participation_cap)),
      start_time_ns_(start_time_ns),
      end_time_ns_(end_time_ns),
      rng_(seed + static_cast<std::uint64_t>(agent_index_) * 104729ULL) {}

int LargeInstitutionalAgent::wake_up(LimitOrderBook& book, std::int64_t current_time_ns) {
    if (!active_ || current_time_ns < start_time_ns_ || current_time_ns > end_time_ns_) return 0;
    if (remaining_quantity_ <= 0) {
        active_ = false;
        return 0;
    }

    const int sampled = sample_child_quantity(side_, rng_);
    const int cap = best_level_participation_cap(book, side_, participation_cap_);
    if (cap <= 0) return 0;

    const int quantity = std::min({child_quantity_, sampled, cap, remaining_quantity_});
    if (quantity <= 0) return 0;

    const int executed = book.submit_market_order(side_, quantity, current_time_ns);
    remaining_quantity_ -= executed;
    if (remaining_quantity_ <= 0) active_ = false;
    return executed;
}

bool LargeInstitutionalAgent::is_finished() const { return !active_ || remaining_quantity_ <= 0; }
int LargeInstitutionalAgent::remaining_quantity() const { return remaining_quantity_; }
