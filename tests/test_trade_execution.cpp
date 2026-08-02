#include "exchange/DistributedLimitOrderBook.hpp"

#include <cassert>

int main() {
    using namespace dlob;

    constexpr BookId book_id = 42;
    DistributedLimitOrderBook book(1, book_id);
    assert(book.book_id() == book_id);

    OrderMessage ask;
    ask.book_id = book_id;
    ask.arrival_time_ns = 10;
    ask.sequence = 100;
    ask.owner_id = make_owner_id(1, 0);
    ask.agent_kind = AgentKind::MarketMaker;
    ask.action = OrderAction::Limit;
    ask.side = Side::Sell;
    ask.quantity = 7;
    ask.price_ticks = 1'001;
    const ApplyResult ask_result = book.apply(ask);
    assert(ask_result.resting_quantity == 7);
    assert(book.take_trades().empty());

    OrderMessage buy;
    buy.book_id = book_id;
    buy.arrival_time_ns = 20;
    buy.sequence = 200;
    buy.owner_id = make_owner_id(2, 0);
    buy.agent_kind = AgentKind::Informed;
    buy.action = OrderAction::Market;
    buy.side = Side::Buy;
    buy.quantity = 4;
    const ApplyResult result = book.apply(buy);
    assert(result.executed_quantity == 4);

    const auto trades = book.take_trades();
    assert(trades.size() == 1);
    const TradeExecution& trade = trades.front();
    assert(trade.book_id == book_id);
    assert(trade.timestamp_ns == buy.arrival_time_ns);
    assert(trade.trade_sequence == 1);
    assert(trade.price_ticks == ask.price_ticks);
    assert(trade.quantity == buy.quantity);
    assert(trade.buyer_owner_id == buy.owner_id);
    assert(trade.seller_owner_id == ask.owner_id);
    assert(trade.buyer_order_sequence == buy.sequence);
    assert(trade.seller_order_sequence == ask.sequence);
    assert(trade.aggressor_side == Side::Buy);
    assert(trade.aggressor_action == OrderAction::Market);
    assert(book.take_trades().empty());

    const auto reports = book.take_reports();
    assert(reports.size() == 2);
    for (const AgentReport& report : reports) assert(report.book_id == book_id);

    const MarketState state = book.state(30, 1'000.0);
    assert(state.book_id == book_id);

    OrderMessage sell;
    sell.book_id = book_id;
    sell.arrival_time_ns = 40;
    sell.sequence = 300;
    sell.owner_id = make_owner_id(3, 0);
    sell.action = OrderAction::Market;
    sell.side = Side::Sell;
    sell.quantity = 2;

    OrderMessage bid = ask;
    bid.arrival_time_ns = 35;
    bid.sequence = 250;
    bid.side = Side::Buy;
    bid.price_ticks = 999;
    bid.quantity = 2;
    book.apply(bid);
    book.take_reports();
    book.apply(sell);

    const auto sell_trades = book.take_trades();
    assert(sell_trades.size() == 1);
    assert(sell_trades[0].trade_sequence == 2);
    assert(sell_trades[0].buyer_owner_id == bid.owner_id);
    assert(sell_trades[0].seller_owner_id == sell.owner_id);
    assert(sell_trades[0].buyer_order_sequence == bid.sequence);
    assert(sell_trades[0].seller_order_sequence == sell.sequence);
    assert(sell_trades[0].aggressor_side == Side::Sell);
    assert(sell_trades[0].aggressor_action == OrderAction::Market);

    return 0;
}
