#pragma once

#include "simulation/MultiAssetTypes.hpp"

#include <cstdint>
#include <optional>

namespace dlob {

// One deterministic value investor per book.  The agent maintains a latent
// fundamental value and supplies counter-cyclical market demand only when the
// quoted midpoint departs from that value by more than a calibrated threshold.
// Its state is rank-independent, so sequential and exact-MPI executions can be
// compared event for event.
class FundamentalValueAgent {
public:
    FundamentalValueAgent(FundamentalValueConfig config,
                          BookId book_id,
                          double initial_fundamental_ticks,
                          std::uint64_t model_seed);

    [[nodiscard]] std::optional<OrderMessage> make_order(
        const MarketState& state,
        std::int64_t decision_time_ns,
        std::uint64_t decision_sequence);

    void on_trade(const TradeExecution& trade);

    [[nodiscard]] BookId book_id() const noexcept { return book_id_; }
    [[nodiscard]] std::int32_t logical_owner_id() const noexcept {
        return owner_id_;
    }
    [[nodiscard]] double fundamental_value_ticks() const noexcept {
        return fundamental_value_ticks_;
    }
    [[nodiscard]] double last_mispricing_bps() const noexcept {
        return last_mispricing_bps_;
    }
    [[nodiscard]] std::int64_t inventory() const noexcept { return inventory_; }
    [[nodiscard]] std::int64_t cash_ticks() const noexcept { return cash_ticks_; }

private:
    void advance_fundamental(std::uint64_t decision_sequence);

    FundamentalValueConfig config_;
    BookId book_id_ = 0;
    std::int32_t owner_id_ = 0;
    std::uint64_t model_seed_ = 0;
    double fundamental_value_ticks_ = 0.0;
    double last_mispricing_bps_ = 0.0;
    std::int64_t inventory_ = 0;
    std::int64_t cash_ticks_ = 0;
};

} // namespace dlob
