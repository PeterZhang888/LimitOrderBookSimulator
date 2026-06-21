#ifndef STRATEGY_ACCOUNT_HPP
#define STRATEGY_ACCOUNT_HPP

#include "Order.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

class StrategyAccount {
public:
    explicit StrategyAccount(
        double price_scale = 10000.0
    )
        : price_scale_(price_scale)
    {}

    void reset() {
        cash_ = 0.0;
        inventory_ = 0;

        last_mark_to_market_value_ = 0.0;
        peak_mark_to_market_value_ = 0.0;
        max_drawdown_ = 0.0;

        max_abs_inventory_ = 0;
        total_bought_quantity_ = 0;
        total_sold_quantity_ = 0;
        total_traded_quantity_ = 0;
        number_of_fills_ = 0;

        has_been_marked_ = false;
    }

    void process_report(
        const ExecutionReport& report
    ) {
        if (report.quantity <= 0) {
            return;
        }

        const double execution_price =
            static_cast<double>(report.price_ticks)
            / price_scale_;

        const double trade_value =
            static_cast<double>(report.quantity)
            * execution_price;

        if (report.side == Side::Buy) {
            cash_ -= trade_value;
            inventory_ += report.quantity;
            total_bought_quantity_ += report.quantity;
        } else {
            cash_ += trade_value;
            inventory_ -= report.quantity;
            total_sold_quantity_ += report.quantity;
        }

        total_traded_quantity_ += report.quantity;
        ++number_of_fills_;

        max_abs_inventory_ =
            std::max(
                max_abs_inventory_,
                std::abs(inventory_)
            );
    }

    void process_reports(
        const std::vector<ExecutionReport>& reports
    ) {
        for (const auto& report : reports) {
            process_report(report);
        }
    }

    double mark_to_market(
        double mid_price_ticks
    ) {
        const double mid_price =
            mid_price_ticks / price_scale_;

        last_mark_to_market_value_ =
            cash_
            + static_cast<double>(inventory_)
            * mid_price;

        if (!has_been_marked_) {
            peak_mark_to_market_value_ =
                last_mark_to_market_value_;

            has_been_marked_ = true;
        } else {
            peak_mark_to_market_value_ =
                std::max(
                    peak_mark_to_market_value_,
                    last_mark_to_market_value_
                );
        }

        const double drawdown =
            peak_mark_to_market_value_
            - last_mark_to_market_value_;

        max_drawdown_ =
            std::max(
                max_drawdown_,
                drawdown
            );

        return last_mark_to_market_value_;
    }

    double final_pnl(
        double final_mid_price_ticks
    ) {
        return mark_to_market(final_mid_price_ticks);
    }

    double cash() const {
        return cash_;
    }

    int inventory() const {
        return inventory_;
    }

    double mark_to_market_value() const {
        return last_mark_to_market_value_;
    }

    double max_drawdown() const {
        return max_drawdown_;
    }

    int max_abs_inventory() const {
        return max_abs_inventory_;
    }

    int total_bought_quantity() const {
        return total_bought_quantity_;
    }

    int total_sold_quantity() const {
        return total_sold_quantity_;
    }

    int total_traded_quantity() const {
        return total_traded_quantity_;
    }

    std::size_t number_of_fills() const {
        return number_of_fills_;
    }

private:
    double price_scale_ = 10000.0;

    double cash_ = 0.0;
    int inventory_ = 0;

    double last_mark_to_market_value_ = 0.0;
    double peak_mark_to_market_value_ = 0.0;
    double max_drawdown_ = 0.0;

    int max_abs_inventory_ = 0;

    int total_bought_quantity_ = 0;
    int total_sold_quantity_ = 0;
    int total_traded_quantity_ = 0;

    std::size_t number_of_fills_ = 0;

    bool has_been_marked_ = false;
};

#endif
