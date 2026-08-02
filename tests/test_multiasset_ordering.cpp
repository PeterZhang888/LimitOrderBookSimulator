// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#include "simulation/MultiAssetTypes.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <queue>
#include <vector>

int main() {
    using namespace dlob;

    const std::array<MultiAssetEventKey, 7> keys{{
        {100, MultiAssetEventPhase::OrderArrival, 1, 20, 4, 2},
        {99, MultiAssetEventPhase::CrossBookReaction, 9, 99, 99, 99},
        {100, MultiAssetEventPhase::AgentDecision, 8, 88, 88, 88},
        {100, MultiAssetEventPhase::OrderArrival, 0, 90, 90, 90},
        {100, MultiAssetEventPhase::OrderArrival, 1, 19, 90, 90},
        {100, MultiAssetEventPhase::OrderArrival, 1, 20, 3, 90},
        {100, MultiAssetEventPhase::OrderArrival, 1, 20, 4, 1},
    }};

    std::array<std::size_t, keys.size()> permutation{};
    std::iota(permutation.begin(), permutation.end(), std::size_t{0});

    std::vector<MultiAssetEventKey> expected(keys.begin(), keys.end());
    std::sort(expected.begin(), expected.end());

    // Exhaust every insertion order (7! = 5040).  The pop order must be a
    // function of the exact key alone, never of insertion or container layout.
    do {
        std::priority_queue<MultiAssetEvent,
                            std::vector<MultiAssetEvent>,
                            MultiAssetEventLater> queue;
        for (const std::size_t index : permutation) {
            MultiAssetEvent event;
            event.key = keys[index];
            queue.push(event);
        }

        for (const MultiAssetEventKey& wanted : expected) {
            assert(!queue.empty());
            assert(queue.top().key == wanted);
            queue.pop();
        }
        assert(queue.empty());
    } while (std::next_permutation(permutation.begin(), permutation.end()));

    // The stable sequence helper has no process/rank argument and is sensitive
    // to every model-level identity component supplied to it.
    const auto base = stable_sequence(shared_market_maker_entity, 42, 0);
    assert(base == stable_sequence(shared_market_maker_entity, 42, 0));
    assert(base != stable_sequence(shared_market_maker_entity, 43, 0));
    assert(base != stable_sequence(shared_market_maker_entity, 42, 1));
    assert(base != stable_sequence(background_entity(1), 42, 0));

    std::cout << "multi-asset exact ordering tests passed\n";
    return 0;
}
