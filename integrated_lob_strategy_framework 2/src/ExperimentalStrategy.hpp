#pragma once

#include "LimitOrderBook.hpp"
#include "Order.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct MarketState {
    std::int64_t timestamp_ns = 0;
    int tick_size = 100;
    int best_bid_ticks = 0;
    int best_ask_ticks = 0;
    double mid_price_ticks = 0.0;
    int spread_ticks = 0;
    int bid_queue_depth = 0;
    int ask_queue_depth = 0;
    double hawkes_total_intensity = 0.0;
    double hawkes_average_intensity = 1.0;
};

struct StrategyQuote {
    bool should_quote = false;
    int bid_price_ticks = 0;
    int ask_price_ticks = 0;
    int bid_volume = 0;
    int ask_volume = 0;
};

struct StrategyDayResult {
    int strategy_id = 0;
    int replica_id = 0;
    int day_id = 0;
    double hyperparameter = 0.0;
    double final_pnl = 0.0;
    int final_inventory = 0;
    double avg_spread = 0.0;
    double total_traded_volume = 0.0;
    double sharpe_ratio = 0.0;
    double max_drawdown = 0.0;
    double mean_abs_position = 0.0;
};

class ExperimentalMarketMaker {
public:
    ExperimentalMarketMaker(int strategy_id, double hyperparameter, int owner_id, int base_order_size, int base_spread_ticks);
    virtual ~ExperimentalMarketMaker() = default;

    virtual StrategyQuote get_quote(const MarketState& state) = 0;
    virtual std::string name() const = 0;

    void process_execution_reports(const std::vector<ExecutionReport>& reports);
    double mark_to_market(double mid_price_ticks) const;
    int inventory() const { return inventory_; }
    double cash_ticks() const { return cash_ticks_; }
    double total_traded_volume() const { return total_traded_volume_; }
    double gross_notional_ticks() const { return gross_notional_ticks_; }
    int owner_id() const { return owner_id_; }
    int strategy_id() const { return strategy_id_; }
    double hyperparameter() const { return hyperparameter_; }

protected:
    int strategy_id_ = 0;
    double hyperparameter_ = 0.0;
    int owner_id_ = 0;
    int base_order_size_ = 100;
    int base_spread_ticks_ = 4;
    int inventory_ = 0;
    double cash_ticks_ = 0.0;
    double total_traded_volume_ = 0.0;
    double gross_notional_ticks_ = 0.0;
};

class InventoryLinearSkewStrategy final : public ExperimentalMarketMaker {
public:
    InventoryLinearSkewStrategy(double gamma, int owner_id, int base_order_size, int base_spread_ticks);
    StrategyQuote get_quote(const MarketState& state) override;
    std::string name() const override { return "InventoryLinearSkew"; }
};

class QueueAwareFillHazardStrategy final : public ExperimentalMarketMaker {
public:
    QueueAwareFillHazardStrategy(double eta, int owner_id, int base_order_size, int base_spread_ticks);
    StrategyQuote get_quote(const MarketState& state) override;
    std::string name() const override { return "QueueAwareFillHazard"; }
};

class HawkesSignalHazardStrategy final : public ExperimentalMarketMaker {
public:
    HawkesSignalHazardStrategy(double theta, int owner_id, int base_order_size, int base_spread_ticks);
    StrategyQuote get_quote(const MarketState& state) override;
    std::string name() const override { return "HawkesSignalHazard"; }
};

std::unique_ptr<ExperimentalMarketMaker> make_strategy(
    int strategy_id,
    double hyperparameter,
    int owner_id,
    int base_order_size,
    int base_spread_ticks
);

std::vector<double> hyperparameter_grid(int strategy_id);
std::string strategy_name(int strategy_id);
