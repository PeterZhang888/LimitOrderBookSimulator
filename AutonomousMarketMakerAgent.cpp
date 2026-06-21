#include "AutonomousMarketMakerAgent.hpp"

#include "Order.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

int AutonomousMarketMakerAgent::number_of_agents_ = 0;

AutonomousMarketMakerAgent::AutonomousMarketMakerAgent(
    int tick_size,
    int base_half_spread_ticks,
    int base_order_quantity,
    int max_inventory,
    double inventory_risk_strength,
    double volatility_strength,
    double signal_strength,
    double aggressive_inventory_ratio,
    std::int64_t refresh_interval_ns
)
    : agent_index_(number_of_agents_),
      owner_id_(2000000 + number_of_agents_),
      tick_size_(
          std::max(1, tick_size)
      ),
      max_inventory_(
          std::max(1, max_inventory)
      ),
      refresh_interval_ns_(
          std::max<std::int64_t>(
              1,
              refresh_interval_ns
          )
      ),
      base_half_spread_ticks_(
          std::max(
              1,
              base_half_spread_ticks
          )
      ),
      base_order_quantity_(
          std::max(
              1,
              base_order_quantity
          )
      ),
      inventory_risk_strength_(
          std::max(
              0.0,
              inventory_risk_strength
          )
      ),
      volatility_strength_(
          std::max(
              0.0,
              volatility_strength
          )
      ),
      signal_strength_(
          std::max(
              0.0,
              signal_strength
          )
      ),
      aggressive_inventory_ratio_(
          std::clamp(
              aggressive_inventory_ratio,
              0.10,
              1.00
          )
      ),
      account_(10000.0)
{
    ++number_of_agents_;
}

bool AutonomousMarketMakerAgent::should_refresh(
    std::int64_t current_time_ns
) const {
    return (
        last_refresh_time_ns_ < 0
        || current_time_ns
            - last_refresh_time_ns_
            >= refresh_interval_ns_
    );
}

int AutonomousMarketMakerAgent::round_to_tick(
    double price
) const {
    const long long rounded =
        static_cast<long long>(
            std::llround(
                price
                / static_cast<double>(
                    tick_size_
                )
            )
        )
        * static_cast<long long>(
            tick_size_
        );

    if (rounded < 1LL) {
        return 1;
    }

    if (
        rounded
        > static_cast<long long>(
            std::numeric_limits<int>::max()
        )
    ) {
        return std::numeric_limits<int>::max();
    }

    return static_cast<int>(rounded);
}

void AutonomousMarketMakerAgent::process_fills(
    LimitOrderBook& book,
    std::int64_t current_time_ns
) {
    (void) current_time_ns;

    account_.process_reports(
        book.get_and_clear_execution_reports(
            owner_id_
        )
    );

    if (book.has_bid() && book.has_ask()) {
        account_.mark_to_market(
            book.mid_price()
        );
    }
}

void AutonomousMarketMakerAgent::update_market_state(
    const LimitOrderBook& book
) {
    if (!book.has_bid() || !book.has_ask()) {
        return;
    }

    mid_price_history_.push_back(
        book.mid_price()
    );

    while (
        mid_price_history_.size()
        > VOLATILITY_WINDOW_SAMPLES
    ) {
        mid_price_history_.pop_front();
    }

    current_volatility_ticks_ =
        calculate_volatility_ticks();

    current_depth_imbalance_ =
        calculate_depth_imbalance(book);

    update_order_flow_imbalance(book);

    current_momentum_signal_ =
        calculate_momentum_signal();

    current_signal_ =
        calculate_combined_signal();
}

double
AutonomousMarketMakerAgent::calculate_volatility_ticks()
const {
    if (mid_price_history_.size() < 3) {
        return 0.0;
    }

    std::vector<double> changes;

    changes.reserve(
        mid_price_history_.size() - 1
    );

    for (
        std::size_t i = 1;
        i < mid_price_history_.size();
        ++i
    ) {
        changes.push_back(
            (
                mid_price_history_[i]
                - mid_price_history_[i - 1]
            )
            / static_cast<double>(
                tick_size_
            )
        );
    }

    double mean = 0.0;

    for (double value : changes) {
        mean += value;
    }

    mean /= static_cast<double>(
        changes.size()
    );

    double squared_sum = 0.0;

    for (double value : changes) {
        const double difference =
            value - mean;

        squared_sum +=
            difference * difference;
    }

    return std::sqrt(
        squared_sum
        / static_cast<double>(
            changes.size() - 1
        )
    );
}

double
AutonomousMarketMakerAgent::calculate_depth_imbalance(
    const LimitOrderBook& book
) const {
    const int bid_quantity =
        book.quantity_at_best_bid();

    const int ask_quantity =
        book.quantity_at_best_ask();

    const long long total =
        static_cast<long long>(
            bid_quantity
        )
        + static_cast<long long>(
            ask_quantity
        );

    if (total <= 0) {
        return 0.0;
    }

    return std::clamp(
        static_cast<double>(
            bid_quantity - ask_quantity
        )
        / static_cast<double>(total),
        -1.0,
        1.0
    );
}

void
AutonomousMarketMakerAgent::update_order_flow_imbalance(
    const LimitOrderBook& book
) {
    const std::uint64_t cumulative_buy =
        book.cumulative_aggressive_buy_quantity();

    const std::uint64_t cumulative_sell =
        book.cumulative_aggressive_sell_quantity();

    if (!flow_counters_initialized_) {
        last_seen_aggressive_buy_quantity_ =
            cumulative_buy;

        last_seen_aggressive_sell_quantity_ =
            cumulative_sell;

        flow_counters_initialized_ = true;

        return;
    }

    const std::uint64_t buy_delta =
        cumulative_buy
        - last_seen_aggressive_buy_quantity_;

    const std::uint64_t sell_delta =
        cumulative_sell
        - last_seen_aggressive_sell_quantity_;

    last_seen_aggressive_buy_quantity_ =
        cumulative_buy;

    last_seen_aggressive_sell_quantity_ =
        cumulative_sell;

    const std::uint64_t total =
        buy_delta + sell_delta;

    double raw_imbalance = 0.0;

    if (total > 0) {
        raw_imbalance =
            (
                static_cast<double>(
                    buy_delta
                )
                - static_cast<double>(
                    sell_delta
                )
            )
            / static_cast<double>(total);
    }

    current_order_flow_imbalance_ =
        (
            1.0
            - ORDER_FLOW_EWMA_WEIGHT
        )
        * current_order_flow_imbalance_
        + ORDER_FLOW_EWMA_WEIGHT
          * std::clamp(
                raw_imbalance,
                -1.0,
                1.0
            );
}

double
AutonomousMarketMakerAgent::calculate_momentum_signal()
const {
    if (mid_price_history_.size() < 2) {
        return 0.0;
    }

    const std::size_t lookback =
        std::min<std::size_t>(
            5,
            mid_price_history_.size() - 1
        );

    const std::size_t start =
        mid_price_history_.size()
        - 1
        - lookback;

    const double change_ticks =
        (
            mid_price_history_.back()
            - mid_price_history_[start]
        )
        / static_cast<double>(
            tick_size_
        );

    return std::tanh(
        change_ticks / 3.0
    );
}

double
AutonomousMarketMakerAgent::calculate_combined_signal()
const {
    // Fixed weights do not add extra parameters.
    //
    // 50% order-book depth
    // 30% aggressive order flow
    // 20% recent momentum
    return std::clamp(
        0.50
            * current_depth_imbalance_
        + 0.30
            * current_order_flow_imbalance_
        + 0.20
            * current_momentum_signal_,
        -1.0,
        1.0
    );
}

int AutonomousMarketMakerAgent::calculate_order_quantity(
    bool is_bid,
    double inventory_ratio,
    double volatility_ticks
) const {
    double inventory_multiplier =
        is_bid
        ? 1.0 - inventory_ratio
        : 1.0 + inventory_ratio;

    inventory_multiplier =
        std::clamp(
            inventory_multiplier,
            0.20,
            1.80
        );

    const double volatility_multiplier =
        1.0
        / (
            1.0
            + 0.25
              * std::max(
                    0.0,
                    volatility_ticks
                )
        );

    return std::max(
        1,
        static_cast<int>(
            std::llround(
                static_cast<double>(
                    base_order_quantity_
                )
                * inventory_multiplier
                * volatility_multiplier
            )
        )
    );
}

void AutonomousMarketMakerAgent::submit_limit_order(
    LimitOrderBook& book,
    std::uint64_t& next_order_id,
    std::int64_t current_time_ns,
    Side side,
    int quantity,
    int price_ticks
) {
    if (
        quantity <= 0
        || price_ticks <= 0
    ) {
        return;
    }

    book.add_limit_order(Order{
        next_order_id++,
        current_time_ns,
        OrderType::Limit,
        side,
        quantity,
        price_ticks,
        owner_id_
    });
}

void
AutonomousMarketMakerAgent::
submit_aggressive_inventory_reduction(
    LimitOrderBook& book,
    std::uint64_t& next_order_id,
    std::int64_t current_time_ns
) {
    const int position =
        account_.inventory();

    if (
        position == 0
        || !book.has_bid()
        || !book.has_ask()
    ) {
        return;
    }

    const int absolute_position =
        std::abs(position);

    const int quantity =
        std::min(
            absolute_position,
            std::max(
                base_order_quantity_,
                absolute_position / 4
            )
        );

    submit_limit_order(
        book,
        next_order_id,
        current_time_ns,
        position > 0
            ? Side::Sell
            : Side::Buy,
        quantity,
        position > 0
            ? book.best_bid()
            : book.best_ask()
    );

    process_fills(
        book,
        current_time_ns
    );

    // Do not treat this agent's own emergency trade as
    // predictive market order flow at the next wake-up.
    last_seen_aggressive_buy_quantity_ =
        book.cumulative_aggressive_buy_quantity();

    last_seen_aggressive_sell_quantity_ =
        book.cumulative_aggressive_sell_quantity();

    ++aggressive_actions_;
}

void AutonomousMarketMakerAgent::wake_up(
    LimitOrderBook& book,
    std::uint64_t& next_order_id,
    std::int64_t current_time_ns
) {
    process_fills(
        book,
        current_time_ns
    );

    update_market_state(book);

    if (!should_refresh(current_time_ns)) {
        return;
    }

    last_refresh_time_ns_ =
        current_time_ns;

    // Cancel-and-requote architecture.
    book.cancel_orders_by_owner(
        owner_id_
    );

    if (!book.has_bid() || !book.has_ask()) {
        return;
    }

    const int best_bid =
        book.best_bid();

    const int best_ask =
        book.best_ask();

    if (best_ask <= best_bid) {
        return;
    }

    const double spread_ticks =
        static_cast<double>(
            best_ask - best_bid
        )
        / static_cast<double>(
            tick_size_
        );

    const double mid_price =
        book.mid_price();

    const int position =
        account_.inventory();

    const double inventory_ratio =
        std::clamp(
            static_cast<double>(position)
            / static_cast<double>(
                max_inventory_
            ),
            -1.0,
            1.0
        );

    if (
        std::abs(inventory_ratio)
        >= aggressive_inventory_ratio_
    ) {
        submit_aggressive_inventory_reduction(
            book,
            next_order_id,
            current_time_ns
        );

        return;
    }

    const double fair_value =
        mid_price
        + signal_strength_
          * current_signal_
          * static_cast<double>(
                tick_size_
            );

    const double reservation_price =
        fair_value
        - inventory_risk_strength_
          * inventory_ratio
          * (
                1.0
                + std::min(
                    current_volatility_ticks_,
                    3.0
                )
            )
          * static_cast<double>(
                tick_size_
            );

    const double half_spread_ticks =
        std::clamp(
            static_cast<double>(
                base_half_spread_ticks_
            )
            + volatility_strength_
              * std::min(
                    current_volatility_ticks_,
                    5.0
                )
            + 0.50
              * std::abs(
                    current_signal_
                ),
            1.0,
            10.0
        );

    int bid_price =
        round_to_tick(
            reservation_price
            - half_spread_ticks
              * static_cast<double>(
                    tick_size_
                )
        );

    int ask_price =
        round_to_tick(
            reservation_price
            + half_spread_ticks
              * static_cast<double>(
                    tick_size_
                )
        );

    bool quote_bid = true;
    bool quote_ask = true;

    const bool high_volatility =
        current_volatility_ticks_
        >= HIGH_VOLATILITY_THRESHOLD_TICKS;

    if (
        high_volatility
        && current_signal_
            >= TOXIC_SIGNAL_THRESHOLD
    ) {
        // Rising-price risk: selling is dangerous.
        quote_ask = false;
        ++defensive_actions_;
    }

    if (
        high_volatility
        && current_signal_
            <= -TOXIC_SIGNAL_THRESHOLD
    ) {
        // Falling-price risk: buying is dangerous.
        quote_bid = false;
        ++defensive_actions_;
    }

    if (inventory_ratio >= 0.60) {
        quote_bid = false;
    }

    if (inventory_ratio <= -0.60) {
        quote_ask = false;
    }

    const int desired_bid_quantity =
        calculate_order_quantity(
            true,
            inventory_ratio,
            current_volatility_ticks_
        );

    const int desired_ask_quantity =
        calculate_order_quantity(
            false,
            inventory_ratio,
            current_volatility_ticks_
        );

    const int bid_capacity =
        std::max(
            0,
            max_inventory_ - position
        );

    const int ask_capacity =
        std::max(
            0,
            max_inventory_ + position
        );

    int bid_quantity_level_1 =
        quote_bid
        ? std::min(
              desired_bid_quantity,
              bid_capacity
          )
        : 0;

    int ask_quantity_level_1 =
        quote_ask
        ? std::min(
              desired_ask_quantity,
              ask_capacity
          )
        : 0;

    int bid_quantity_level_2 = 0;
    int ask_quantity_level_2 = 0;

    if (bid_quantity_level_1 > 0) {
        bid_quantity_level_2 =
            std::min(
                std::max(
                    1,
                    static_cast<int>(
                        std::llround(
                            static_cast<double>(
                                bid_quantity_level_1
                            )
                            * SECOND_LEVEL_QUANTITY_RATIO
                        )
                    )
                ),
                bid_capacity
                - bid_quantity_level_1
            );
    }

    if (ask_quantity_level_1 > 0) {
        ask_quantity_level_2 =
            std::min(
                std::max(
                    1,
                    static_cast<int>(
                        std::llround(
                            static_cast<double>(
                                ask_quantity_level_1
                            )
                            * SECOND_LEVEL_QUANTITY_RATIO
                        )
                    )
                ),
                ask_capacity
                - ask_quantity_level_1
            );
    }

    quote_bid =
        quote_bid
        && bid_quantity_level_1 > 0;

    quote_ask =
        quote_ask
        && ask_quantity_level_1 > 0;

    // Join the best quote when the model price is nearby.
    if (
        quote_bid
        && bid_price
            >= best_bid - tick_size_
    ) {
        bid_price = best_bid;
    }

    if (
        quote_ask
        && ask_price
            <= best_ask + tick_size_
    ) {
        ask_price = best_ask;
    }

    // Queue-aware competition:
    // improve by one tick only when joining would give the
    // agent a very small fraction of the current queue.
    if (
        quote_bid
        && bid_price == best_bid
    ) {
        const double queue_share =
            static_cast<double>(
                bid_quantity_level_1
            )
            / static_cast<double>(
                book.quantity_at_best_bid()
                + bid_quantity_level_1
            );

        if (
            queue_share
                < MIN_QUEUE_SHARE_TO_JOIN
            && spread_ticks >= 3.0
            && current_signal_ >= -0.25
            && !high_volatility
        ) {
            bid_price =
                best_bid + tick_size_;
        }
    }

    if (
        quote_ask
        && ask_price == best_ask
    ) {
        const double queue_share =
            static_cast<double>(
                ask_quantity_level_1
            )
            / static_cast<double>(
                book.quantity_at_best_ask()
                + ask_quantity_level_1
            );

        if (
            queue_share
                < MIN_QUEUE_SHARE_TO_JOIN
            && spread_ticks >= 3.0
            && current_signal_ <= 0.25
            && !high_volatility
        ) {
            ask_price =
                best_ask - tick_size_;
        }
    }

    // Normal and defensive modes remain passive.
    bid_price =
        std::min(
            bid_price,
            best_ask - tick_size_
        );

    ask_price =
        std::max(
            ask_price,
            best_bid + tick_size_
        );

    if (
        bid_price <= 0
        || ask_price <= 0
    ) {
        return;
    }

    if (
        quote_bid
        && quote_ask
        && bid_price >= ask_price
    ) {
        return;
    }

    // First level.
    if (quote_bid) {
        submit_limit_order(
            book,
            next_order_id,
            current_time_ns,
            Side::Buy,
            bid_quantity_level_1,
            bid_price
        );

        // Second bid level: one tick deeper,
        // with 50% of the first-level quantity.
        if (bid_quantity_level_2 > 0) {
            submit_limit_order(
                book,
                next_order_id,
                current_time_ns,
                Side::Buy,
                bid_quantity_level_2,
                bid_price - tick_size_
            );
        }
    }

    if (quote_ask) {
        submit_limit_order(
            book,
            next_order_id,
            current_time_ns,
            Side::Sell,
            ask_quantity_level_1,
            ask_price
        );

        // Second ask level: one tick deeper,
        // with 50% of the first-level quantity.
        if (ask_quantity_level_2 > 0) {
            submit_limit_order(
                book,
                next_order_id,
                current_time_ns,
                Side::Sell,
                ask_quantity_level_2,
                ask_price + tick_size_
            );
        }
    }
}

double AutonomousMarketMakerAgent::mark_to_market(
    const LimitOrderBook& book
) {
    if (!book.has_bid() || !book.has_ask()) {
        return account_.mark_to_market_value();
    }

    return account_.mark_to_market(
        book.mid_price()
    );
}

double AutonomousMarketMakerAgent::final_pnl(
    const LimitOrderBook& book
) {
    return mark_to_market(book);
}

int AutonomousMarketMakerAgent::agent_index() const {
    return agent_index_;
}

int AutonomousMarketMakerAgent::owner_id() const {
    return owner_id_;
}

double AutonomousMarketMakerAgent::cash() const {
    return account_.cash();
}

int AutonomousMarketMakerAgent::inventory() const {
    return account_.inventory();
}

double
AutonomousMarketMakerAgent::mark_to_market_value()
const {
    return account_.mark_to_market_value();
}

double AutonomousMarketMakerAgent::max_drawdown()
const {
    return account_.max_drawdown();
}

int AutonomousMarketMakerAgent::max_abs_inventory()
const {
    return account_.max_abs_inventory();
}

int
AutonomousMarketMakerAgent::total_traded_quantity()
const {
    return account_.total_traded_quantity();
}

std::size_t
AutonomousMarketMakerAgent::number_of_fills()
const {
    return account_.number_of_fills();
}

double
AutonomousMarketMakerAgent::current_volatility_ticks()
const {
    return current_volatility_ticks_;
}

double AutonomousMarketMakerAgent::current_signal()
const {
    return current_signal_;
}

double
AutonomousMarketMakerAgent::current_depth_imbalance()
const {
    return current_depth_imbalance_;
}

double
AutonomousMarketMakerAgent::current_order_flow_imbalance()
const {
    return current_order_flow_imbalance_;
}

double
AutonomousMarketMakerAgent::current_momentum_signal()
const {
    return current_momentum_signal_;
}

std::size_t
AutonomousMarketMakerAgent::defensive_actions()
const {
    return defensive_actions_;
}

std::size_t
AutonomousMarketMakerAgent::aggressive_actions()
const {
    return aggressive_actions_;
}

int AutonomousMarketMakerAgent::number_of_agents() {
    return number_of_agents_;
}
