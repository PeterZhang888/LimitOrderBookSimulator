// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#pragma once

#include "simulation/MultiAssetTypes.hpp"

#include <cstdint>
#include <vector>

namespace dlob {

// Deterministic reduced-basket arbitrage agent.  It compares normalized ETF
// price with a configurable weighted component basket and emits one market
// order per leg only when the threshold state changes.  The hysteresis avoids
// repeatedly trading the same persistent signal at every observation.
class EtfArbitrageAgent {
public:
    EtfArbitrageAgent(EtfArbitrageConfig config,
                      std::vector<MultiAssetBookConfig> books);

    [[nodiscard]] std::vector<OrderMessage> make_orders(
        const std::vector<MarketState>& states,
        std::int64_t decision_time_ns,
        std::uint64_t decision_sequence);

    void on_trade(const TradeExecution& trade);
    [[nodiscard]] std::int64_t inventory(BookId book_id) const;
    [[nodiscard]] std::int64_t cash_ticks(BookId book_id) const;
    [[nodiscard]] std::int64_t total_cash_ticks() const;

    [[nodiscard]] double last_deviation_bps() const noexcept {
        return last_deviation_bps_;
    }

private:
    EtfArbitrageConfig config_;
    std::vector<MultiAssetBookConfig> books_;
    std::vector<double> normalized_weights_;
    std::vector<std::int64_t> inventory_by_book_;
    std::vector<std::int64_t> cash_by_book_;
    int last_signal_ = 0;
    double last_deviation_bps_ = 0.0;
};

} // namespace dlob
