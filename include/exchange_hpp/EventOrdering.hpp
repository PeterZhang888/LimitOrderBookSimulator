#pragma once

#include "common/DistributedTypes.hpp"

namespace dlob {

inline int same_owner_action_priority(OrderAction action) {
    // Cancel-and-replace messages from one market maker share an arrival time.
    // The old quotes must be removed before replacement limits are inserted.
    return action == OrderAction::CancelOwner ? 0 : 1;
}

inline bool order_before(const OrderMessage& left, const OrderMessage& right) {
    if (left.arrival_time_ns != right.arrival_time_ns) {
        return left.arrival_time_ns < right.arrival_time_ns;
    }

    // Preserve a deterministic causal order for a single agent's batch.
    if (left.owner_id > 0 && left.owner_id == right.owner_id
        && left.generated_time_ns == right.generated_time_ns) {
        const int left_priority = same_owner_action_priority(left.action);
        const int right_priority = same_owner_action_priority(right.action);
        if (left_priority != right_priority) return left_priority < right_priority;
        if (left.sequence != right.sequence) return left.sequence < right.sequence;
    }

    // For genuinely concurrent messages from different agents, use the seeded
    // tie-breaker before rank so MPI partitioning cannot create priority bias.
    if (left.tie_breaker != right.tie_breaker) return left.tie_breaker < right.tie_breaker;
    if (left.source_rank != right.source_rank) return left.source_rank < right.source_rank;
    return left.sequence < right.sequence;
}

} // namespace dlob
