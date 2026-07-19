#ifndef AUTONOMOUS_MARKET_MAKER_AGENT_HPP
#define AUTONOMOUS_MARKET_MAKER_AGENT_HPP

#include "LimitOrderBook.hpp"
#include "StrategyAccount.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>

class AutonomousMarketMakerAgent {
public:
    AutonomousMarketMakerAgent(
        int tick_size,
        int base_half_spread_ticks,
        int base_order_quantity,
        int max_inventory,
        double inventory_risk_strength,
        double volatility_strength,
        double signal_strength,
        double aggressive_inventory_ratio,
        std::int64_t refresh_interval_ns
    );

    AutonomousMarketMakerAgent(
        const AutonomousMarketMakerAgent&
    ) = delete;

    AutonomousMarketMakerAgent& operator=(
        const AutonomousMarketMakerAgent&
    ) = delete;

    AutonomousMarketMakerAgent(
        AutonomousMarketMakerAgent&&
    ) = default;

    AutonomousMarketMakerAgent& operator=(
        AutonomousMarketMakerAgent&&
    ) = default;

    void wake_up(
        LimitOrderBook& book,
        std::uint64_t& next_order_id,
        std::int64_t current_time_ns
    );

    void process_fills(
        LimitOrderBook& book,
        std::int64_t current_time_ns
    );

    double mark_to_market(
        const LimitOrderBook& book
    );

    double final_pnl(
        const LimitOrderBook& book
    );

    int agent_index() const;
    int owner_id() const;

    double cash() const;
    int inventory() const;
    double mark_to_market_value() const;
    double max_drawdown() const;

    int max_abs_inventory() const;
    int total_traded_quantity() const;

    std::size_t number_of_fills() const;

    double current_volatility_ticks() const;
    double current_signal() const;
    double current_depth_imbalance() const;
    double current_order_flow_imbalance() const;
    double current_momentum_signal() const;

    std::size_t defensive_actions() const;
    std::size_t aggressive_actions() const;

    static int number_of_agents();

private:
    int agent_index_ = 0;
    int owner_id_ = -1;

    int tick_size_ = 100;
    int max_inventory_ = 500;

    std::int64_t refresh_interval_ns_ =
        1000LL * 1000000LL;

    std::int64_t last_refresh_time_ns_ = -1;

    // Six tunable strategy parameters.
    int base_half_spread_ticks_ = 2;
    int base_order_quantity_ = 10;

    double inventory_risk_strength_ = 2.0;
    double volatility_strength_ = 0.5;
    double signal_strength_ = 0.5;
    double aggressive_inventory_ratio_ = 0.80;

    // Fixed algorithm-design values.
    static constexpr std::size_t
        VOLATILITY_WINDOW_SAMPLES = 30;

    static constexpr double
        ORDER_FLOW_EWMA_WEIGHT = 0.20;

    static constexpr double
        TOXIC_SIGNAL_THRESHOLD = 0.65;

    static constexpr double
        HIGH_VOLATILITY_THRESHOLD_TICKS = 1.50;

    static constexpr double
        MIN_QUEUE_SHARE_TO_JOIN = 0.10;

    static constexpr double
        SECOND_LEVEL_QUANTITY_RATIO = 0.50;

    std::deque<double> mid_price_history_;

    double current_volatility_ticks_ = 0.0;
    double current_depth_imbalance_ = 0.0;
    double current_order_flow_imbalance_ = 0.0;
    double current_momentum_signal_ = 0.0;
    double current_signal_ = 0.0;

    bool flow_counters_initialized_ = false;

    std::uint64_t
        last_seen_aggressive_buy_quantity_ = 0;

    std::uint64_t
        last_seen_aggressive_sell_quantity_ = 0;

    std::size_t defensive_actions_ = 0;
    std::size_t aggressive_actions_ = 0;

    StrategyAccount account_;

    bool should_refresh(
        std::int64_t current_time_ns
    ) const;

    int round_to_tick(
        double price
    ) const;

    void update_market_state(
        const LimitOrderBook& book
    );

    double calculate_volatility_ticks() const;

    double calculate_depth_imbalance(
        const LimitOrderBook& book
    ) const;

    void update_order_flow_imbalance(
        const LimitOrderBook& book
    );

    double calculate_momentum_signal() const;

    double calculate_combined_signal() const;

    int calculate_order_quantity(
        bool is_bid,
        double inventory_ratio,
        double volatility_ticks
    ) const;

    void submit_limit_order(
        LimitOrderBook& book,
        std::uint64_t& next_order_id,
        std::int64_t current_time_ns,
        Side side,
        int quantity,
        int price_ticks
    );

    void submit_aggressive_inventory_reduction(
        LimitOrderBook& book,
        std::uint64_t& next_order_id,
        std::int64_t current_time_ns
    );

    static int number_of_agents_;
};

#endif
