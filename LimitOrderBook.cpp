#include "LimitOrderBook.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

// ============================================================
// Tracking helpers
// ============================================================

void LimitOrderBook::remove_order_from_tracking(
    const Order& order
) {
    order_index_.erase(order.id);

    if (order.owner_id < 0) {
        return;
    }

    auto owner_it =
        owner_order_ids_.find(order.owner_id);

    if (owner_it == owner_order_ids_.end()) {
        return;
    }

    owner_it->second.erase(order.id);

    if (owner_it->second.empty()) {
        owner_order_ids_.erase(owner_it);
    }
}

void LimitOrderBook::record_execution_report(
    std::int64_t timestamp_ns,
    const Order& order,
    int executed_quantity,
    int execution_price_ticks,
    bool liquidity_provider
) {
    if (order.owner_id < 0) {
        return;
    }

    if (executed_quantity <= 0) {
        return;
    }

    execution_reports_.push_back(
        ExecutionReport{
            timestamp_ns,
            order.owner_id,
            order.side,
            executed_quantity,
            execution_price_ticks,
            liquidity_provider
        }
    );
}

void LimitOrderBook::record_aggressive_execution(
    Side aggressive_side,
    int executed_quantity
) {
    if (executed_quantity <= 0) {
        return;
    }

    if (aggressive_side == Side::Buy) {
        cumulative_aggressive_buy_quantity_ +=
            static_cast<std::uint64_t>(
                executed_quantity
            );

        return;
    }

    cumulative_aggressive_sell_quantity_ +=
        static_cast<std::uint64_t>(
            executed_quantity
        );
}

int LimitOrderBook::queue_quantity(
    const OrderQueue& queue
) {
    long long total = 0;

    for (const Order& order : queue) {
        total += order.quantity;
    }

    if (
        total
        > static_cast<long long>(
            std::numeric_limits<int>::max()
        )
    ) {
        return std::numeric_limits<int>::max();
    }

    return static_cast<int>(total);
}

// ============================================================
// Execution-report access
// ============================================================

const std::vector<ExecutionReport>&
LimitOrderBook::execution_reports() const {
    return execution_reports_;
}

void LimitOrderBook::clear_execution_reports() {
    execution_reports_.clear();
}

std::vector<ExecutionReport>
LimitOrderBook::get_and_clear_execution_reports() {
    std::vector<ExecutionReport> reports;

    reports.swap(execution_reports_);

    return reports;
}

std::vector<ExecutionReport>
LimitOrderBook::get_and_clear_execution_reports(
    int owner_id
) {
    std::vector<ExecutionReport>
        selected_reports;

    std::vector<ExecutionReport>
        remaining_reports;

    selected_reports.reserve(
        execution_reports_.size()
    );

    remaining_reports.reserve(
        execution_reports_.size()
    );

    for (
        const ExecutionReport& report :
        execution_reports_
    ) {
        if (report.owner_id == owner_id) {
            selected_reports.push_back(report);
        } else {
            remaining_reports.push_back(report);
        }
    }

    execution_reports_.swap(
        remaining_reports
    );

    return selected_reports;
}

// ============================================================
// Add limit order
// ============================================================

void LimitOrderBook::add_limit_order(
    const Order& input_order
) {
    if (input_order.quantity <= 0) {
        return;
    }

    if (input_order.price_ticks <= 0) {
        return;
    }

    // If the ID already exists, remove the old order first.
    if (
        order_index_.find(input_order.id)
        != order_index_.end()
    ) {
        cancel_order(input_order.id);
    }

    Order remaining_order =
        input_order;

    // ========================================================
    // Incoming buy limit order
    // ========================================================

    if (remaining_order.side == Side::Buy) {
        // Execute while the best ask is at or below the
        // incoming buy limit price.
        while (
            remaining_order.quantity > 0
            && !asks_.empty()
            && asks_.begin()->first
                <= remaining_order.price_ticks
        ) {
            auto level_it =
                asks_.begin();

            const int execution_price =
                level_it->first;

            OrderQueue& queue =
                level_it->second;

            while (
                remaining_order.quantity > 0
                && !queue.empty()
            ) {
                auto resting_it =
                    queue.begin();

                const int executed =
                    std::min(
                        remaining_order.quantity,
                        resting_it->quantity
                    );

                // The resting sell owner provided liquidity.
                record_execution_report(
                    remaining_order.timestamp_ns,
                    *resting_it,
                    executed,
                    execution_price,
                    true
                );

                // The incoming buy owner took liquidity.
                record_execution_report(
                    remaining_order.timestamp_ns,
                    remaining_order,
                    executed,
                    execution_price,
                    false
                );

                record_aggressive_execution(
                    Side::Buy,
                    executed
                );

                remaining_order.quantity -=
                    executed;

                resting_it->quantity -=
                    executed;

                if (resting_it->quantity == 0) {
                    const Order completed_order =
                        *resting_it;

                    remove_order_from_tracking(
                        completed_order
                    );

                    queue.erase(resting_it);
                }
            }

            if (queue.empty()) {
                asks_.erase(level_it);
            }
        }

        // Fully executed: nothing remains to rest.
        if (remaining_order.quantity <= 0) {
            return;
        }

        // Rest the unexecuted quantity on the bid side.
        OrderQueue& queue =
            bids_[remaining_order.price_ticks];

        queue.push_back(remaining_order);

        const auto order_it =
            std::prev(queue.end());

        order_index_.insert_or_assign(
            remaining_order.id,
            OrderLocation{
                Side::Buy,
                remaining_order.price_ticks,
                order_it
            }
        );

        if (remaining_order.owner_id >= 0) {
            owner_order_ids_[
                remaining_order.owner_id
            ].insert(
                remaining_order.id
            );
        }

        return;
    }

    // ========================================================
    // Incoming sell limit order
    // ========================================================

    // Execute while the best bid is at or above the incoming
    // sell limit price.
    while (
        remaining_order.quantity > 0
        && !bids_.empty()
        && bids_.begin()->first
            >= remaining_order.price_ticks
    ) {
        auto level_it =
            bids_.begin();

        const int execution_price =
            level_it->first;

        OrderQueue& queue =
            level_it->second;

        while (
            remaining_order.quantity > 0
            && !queue.empty()
        ) {
            auto resting_it =
                queue.begin();

            const int executed =
                std::min(
                    remaining_order.quantity,
                    resting_it->quantity
                );

            // The resting buy owner provided liquidity.
            record_execution_report(
                remaining_order.timestamp_ns,
                *resting_it,
                executed,
                execution_price,
                true
            );

            // The incoming sell owner took liquidity.
            record_execution_report(
                remaining_order.timestamp_ns,
                remaining_order,
                executed,
                execution_price,
                false
            );

            record_aggressive_execution(
                Side::Sell,
                executed
            );

            remaining_order.quantity -=
                executed;

            resting_it->quantity -=
                executed;

            if (resting_it->quantity == 0) {
                const Order completed_order =
                    *resting_it;

                remove_order_from_tracking(
                    completed_order
                );

                queue.erase(resting_it);
            }
        }

        if (queue.empty()) {
            bids_.erase(level_it);
        }
    }

    // Fully executed.
    if (remaining_order.quantity <= 0) {
        return;
    }

    // Rest the remaining sell quantity.
    OrderQueue& queue =
        asks_[remaining_order.price_ticks];

    queue.push_back(remaining_order);

    const auto order_it =
        std::prev(queue.end());

    order_index_.insert_or_assign(
        remaining_order.id,
        OrderLocation{
            Side::Sell,
            remaining_order.price_ticks,
            order_it
        }
    );

    if (remaining_order.owner_id >= 0) {
        owner_order_ids_[
            remaining_order.owner_id
        ].insert(
            remaining_order.id
        );
    }
}

// ============================================================
// Market orders
// ============================================================

int LimitOrderBook::submit_market_order(
    Side side,
    int quantity,
    std::int64_t timestamp_ns
) {
    if (quantity <= 0) {
        return 0;
    }

    int executed_total = 0;

    // ========================================================
    // Buy market order consumes asks
    // ========================================================

    if (side == Side::Buy) {
        while (
            quantity > 0
            && !asks_.empty()
        ) {
            auto level_it =
                asks_.begin();

            const int execution_price =
                level_it->first;

            OrderQueue& queue =
                level_it->second;

            while (
                quantity > 0
                && !queue.empty()
            ) {
                auto order_it =
                    queue.begin();

                const int executed =
                    std::min(
                        quantity,
                        order_it->quantity
                    );

                // Only the resting order has an owner record.
                record_execution_report(
                    timestamp_ns,
                    *order_it,
                    executed,
                    execution_price,
                    true
                );

                quantity -= executed;

                executed_total += executed;

                order_it->quantity -=
                    executed;

                if (order_it->quantity == 0) {
                    const Order completed_order =
                        *order_it;

                    remove_order_from_tracking(
                        completed_order
                    );

                    queue.erase(order_it);
                }
            }

            if (queue.empty()) {
                asks_.erase(level_it);
            }
        }

        record_aggressive_execution(
            Side::Buy,
            executed_total
        );

        return executed_total;
    }

    // ========================================================
    // Sell market order consumes bids
    // ========================================================

    while (
        quantity > 0
        && !bids_.empty()
    ) {
        auto level_it =
            bids_.begin();

        const int execution_price =
            level_it->first;

        OrderQueue& queue =
            level_it->second;

        while (
            quantity > 0
            && !queue.empty()
        ) {
            auto order_it =
                queue.begin();

            const int executed =
                std::min(
                    quantity,
                    order_it->quantity
                );

            record_execution_report(
                timestamp_ns,
                *order_it,
                executed,
                execution_price,
                true
            );

            quantity -= executed;

            executed_total += executed;

            order_it->quantity -=
                executed;

            if (order_it->quantity == 0) {
                const Order completed_order =
                    *order_it;

                remove_order_from_tracking(
                    completed_order
                );

                queue.erase(order_it);
            }
        }

        if (queue.empty()) {
            bids_.erase(level_it);
        }
    }

    record_aggressive_execution(
        Side::Sell,
        executed_total
    );

    return executed_total;
}

// ============================================================
// Cancel one order
// ============================================================

bool LimitOrderBook::cancel_order(
    std::uint64_t order_id
) {
    auto index_it =
        order_index_.find(order_id);

    if (index_it == order_index_.end()) {
        return false;
    }

    const OrderLocation location =
        index_it->second;

    if (location.side == Side::Buy) {
        auto level_it =
            bids_.find(location.price_ticks);

        if (level_it == bids_.end()) {
            order_index_.erase(index_it);
            return false;
        }

        const Order removed_order =
            *(location.iterator);

        remove_order_from_tracking(
            removed_order
        );

        level_it->second.erase(
            location.iterator
        );

        if (level_it->second.empty()) {
            bids_.erase(level_it);
        }

        return true;
    }

    auto level_it =
        asks_.find(location.price_ticks);

    if (level_it == asks_.end()) {
        order_index_.erase(index_it);
        return false;
    }

    const Order removed_order =
        *(location.iterator);

    remove_order_from_tracking(
        removed_order
    );

    level_it->second.erase(
        location.iterator
    );

    if (level_it->second.empty()) {
        asks_.erase(level_it);
    }

    return true;
}

// ============================================================
// Cancel all orders belonging to one owner
// ============================================================

std::size_t LimitOrderBook::cancel_orders_by_owner(
    int owner_id
) {
    const auto owner_it =
        owner_order_ids_.find(owner_id);

    if (owner_it == owner_order_ids_.end()) {
        return 0;
    }

    // Copy the IDs because cancel_order() modifies the owner set.
    const std::vector<std::uint64_t> order_ids(
        owner_it->second.begin(),
        owner_it->second.end()
    );

    std::size_t cancelled = 0;

    for (
        const std::uint64_t order_id :
        order_ids
    ) {
        if (cancel_order(order_id)) {
            ++cancelled;
        }
    }

    return cancelled;
}

// ============================================================
// Quantity cancellation at one price
// ============================================================

int LimitOrderBook::cancel_at_price(
    Side side,
    int price_ticks,
    int quantity
) {
    if (quantity <= 0) {
        return 0;
    }

    int cancelled_total = 0;

    if (side == Side::Buy) {
        auto level_it =
            bids_.find(price_ticks);

        if (level_it == bids_.end()) {
            return 0;
        }

        OrderQueue& queue =
            level_it->second;

        while (
            quantity > 0
            && !queue.empty()
        ) {
            auto order_it =
                queue.begin();

            const int cancelled =
                std::min(
                    quantity,
                    order_it->quantity
                );

            quantity -= cancelled;

            cancelled_total +=
                cancelled;

            order_it->quantity -=
                cancelled;

            if (order_it->quantity == 0) {
                const Order removed_order =
                    *order_it;

                remove_order_from_tracking(
                    removed_order
                );

                queue.erase(order_it);
            }
        }

        if (queue.empty()) {
            bids_.erase(level_it);
        }

        return cancelled_total;
    }

    auto level_it =
        asks_.find(price_ticks);

    if (level_it == asks_.end()) {
        return 0;
    }

    OrderQueue& queue =
        level_it->second;

    while (
        quantity > 0
        && !queue.empty()
    ) {
        auto order_it =
            queue.begin();

        const int cancelled =
            std::min(
                quantity,
                order_it->quantity
            );

        quantity -= cancelled;

        cancelled_total +=
            cancelled;

        order_it->quantity -=
            cancelled;

        if (order_it->quantity == 0) {
            const Order removed_order =
                *order_it;

            remove_order_from_tracking(
                removed_order
            );

            queue.erase(order_it);
        }
    }

    if (queue.empty()) {
        asks_.erase(level_it);
    }

    return cancelled_total;
}

// ============================================================
// Cancellation by distance from current best price
// ============================================================

void LimitOrderBook::cancel_at_distance(
    Side side,
    int distance_ticks,
    int quantity,
    int tick_size
) {
    if (
        quantity <= 0
        || tick_size <= 0
    ) {
        return;
    }

    if (side == Side::Buy) {
        if (!has_bid()) {
            return;
        }

        const long long target_price =
            static_cast<long long>(
                best_bid()
            )
            - static_cast<long long>(
                distance_ticks
            )
              * static_cast<long long>(
                    tick_size
                );

        if (
            target_price <= 0
            || target_price
                > std::numeric_limits<int>::max()
        ) {
            return;
        }

        cancel_at_price(
            Side::Buy,
            static_cast<int>(target_price),
            quantity
        );

        return;
    }

    if (!has_ask()) {
        return;
    }

    const long long target_price =
        static_cast<long long>(
            best_ask()
        )
        + static_cast<long long>(
            distance_ticks
        )
          * static_cast<long long>(
                tick_size
            );

    if (
        target_price <= 0
        || target_price
            > std::numeric_limits<int>::max()
    ) {
        return;
    }

    cancel_at_price(
        Side::Sell,
        static_cast<int>(target_price),
        quantity
    );
}

// ============================================================
// Cancel quantity from the current best side
// ============================================================

void LimitOrderBook::cancel_from_side(
    Side side,
    int quantity
) {
    if (quantity <= 0) {
        return;
    }

    if (side == Side::Buy) {
        while (
            quantity > 0
            && !bids_.empty()
        ) {
            const int price =
                bids_.begin()->first;

            const int cancelled =
                cancel_at_price(
                    Side::Buy,
                    price,
                    quantity
                );

            if (cancelled <= 0) {
                break;
            }

            quantity -= cancelled;
        }

        return;
    }

    while (
        quantity > 0
        && !asks_.empty()
    ) {
        const int price =
            asks_.begin()->first;

        const int cancelled =
            cancel_at_price(
                Side::Sell,
                price,
                quantity
            );

        if (cancelled <= 0) {
            break;
        }

        quantity -= cancelled;
    }
}

// ============================================================
// Book state
// ============================================================

bool LimitOrderBook::has_bid() const {
    return !bids_.empty();
}

bool LimitOrderBook::has_ask() const {
    return !asks_.empty();
}

int LimitOrderBook::best_bid() const {
    if (bids_.empty()) {
        throw std::runtime_error(
            "Cannot request best bid from an empty bid book."
        );
    }

    return bids_.begin()->first;
}

int LimitOrderBook::best_ask() const {
    if (asks_.empty()) {
        throw std::runtime_error(
            "Cannot request best ask from an empty ask book."
        );
    }

    return asks_.begin()->first;
}

int LimitOrderBook::spread() const {
    if (!has_bid() || !has_ask()) {
        return 0;
    }

    return best_ask() - best_bid();
}

double LimitOrderBook::mid_price() const {
    if (!has_bid() || !has_ask()) {
        return 0.0;
    }

    return 0.5
        * static_cast<double>(
            best_bid() + best_ask()
        );
}

int LimitOrderBook::quantity_at_best_bid() const {
    if (bids_.empty()) {
        return 0;
    }

    return queue_quantity(
        bids_.begin()->second
    );
}

int LimitOrderBook::quantity_at_best_ask() const {
    if (asks_.empty()) {
        return 0;
    }

    return queue_quantity(
        asks_.begin()->second
    );
}

// ============================================================
// Aggressive order-flow counters
// ============================================================

std::uint64_t
LimitOrderBook::cumulative_aggressive_buy_quantity()
const {
    return cumulative_aggressive_buy_quantity_;
}

std::uint64_t
LimitOrderBook::cumulative_aggressive_sell_quantity()
const {
    return cumulative_aggressive_sell_quantity_;
}

// ============================================================
// Debug printing
// ============================================================

void LimitOrderBook::print_book() const {
    std::cout << "ASKS\n";

    for (const auto& level : asks_) {
        std::cout
            << "price=" << level.first
            << " quantity="
            << queue_quantity(level.second)
            << " orders="
            << level.second.size()
            << "\n";
    }

    std::cout << "BIDS\n";

    for (const auto& level : bids_) {
        std::cout
            << "price=" << level.first
            << " quantity="
            << queue_quantity(level.second)
            << " orders="
            << level.second.size()
            << "\n";
    }
}
