#include "common/TradeTapeHasher.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <span>

int main() {
    using namespace dlob;

    std::array<TradeExecution, 2> tape{};
    tape[0].book_id = 7;
    tape[0].timestamp_ns = 34'200'000'000'123LL;
    tape[0].trade_sequence = 1;
    tape[0].price_ticks = 2'203'700;
    tape[0].quantity = 125;
    tape[0].buyer_owner_id = 2'000'001;
    tape[0].seller_owner_id = 1'000'001;
    tape[0].buyer_order_sequence = 0x0200000000000001ULL;
    tape[0].seller_order_sequence = 0x0100000000000002ULL;
    tape[0].aggressor_side = Side::Buy;
    tape[0].aggressor_action = OrderAction::Market;

    tape[1] = tape[0];
    tape[1].timestamp_ns += 17;
    tape[1].trade_sequence = 2;
    tape[1].price_ticks = 2'203'600;
    tape[1].quantity = 25;
    tape[1].buyer_owner_id = 3'000'001;
    tape[1].buyer_order_sequence = 0x0300000000000001ULL;
    tape[1].aggressor_side = Side::Sell;
    tape[1].aggressor_action = OrderAction::Limit;

    TradeTapeHasher incremental;
    incremental.add(tape[0]);
    incremental.update(tape[1]);
    assert(incremental.trade_count() == tape.size());
    assert(incremental.digest() == TradeTapeHasher::hash(tape));
    assert(incremental.value() == hash_trade_tape(tape));

    // This golden value makes the explicit field order and byte order part of
    // the test contract, independent of structure padding on the host.
    assert(incremental.digest() == 990980625659194846ULL);

    auto changed = tape;
    changed[1].quantity += 1;
    assert(hash_trade_tape(changed) != hash_trade_tape(tape));

    changed = tape;
    changed[0].book_id += 1;
    assert(hash_trade_tape(changed) != hash_trade_tape(tape));

    const std::span<const TradeExecution> empty;
    assert(hash_trade_tape(empty) == TradeTapeHasher::offset_basis);

    return 0;
}
