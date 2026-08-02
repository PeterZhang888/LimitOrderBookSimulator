#include "exchange/EventOrdering.hpp"

#include <cassert>
#include <iostream>

int main() {
    using namespace dlob;

    OrderMessage cancel;
    cancel.arrival_time_ns = 100;
    cancel.generated_time_ns = 50;
    cancel.owner_id = 1'000'001;
    cancel.source_rank = 1;
    cancel.action = OrderAction::CancelOwner;
    cancel.sequence = 20;
    cancel.tie_breaker = 999;

    OrderMessage replacement = cancel;
    replacement.action = OrderAction::Limit;
    replacement.sequence = 10;
    replacement.tie_breaker = 1;

    assert(order_before(cancel, replacement));
    assert(!order_before(replacement, cancel));

    OrderMessage other = replacement;
    other.owner_id = 2'000'001;
    other.source_rank = 2;
    other.tie_breaker = 0;
    assert(order_before(other, replacement));

    OrderMessage earlier = replacement;
    earlier.arrival_time_ns = 99;
    assert(order_before(earlier, replacement));

    std::cout << "event ordering tests passed\n";
    return 0;
}
