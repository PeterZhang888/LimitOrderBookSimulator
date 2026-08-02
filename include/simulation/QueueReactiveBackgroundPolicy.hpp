#pragma once

#include "exchange/BackgroundHawkesAgent.hpp"
#include "simulation/MultiAssetTypes.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace dlob {

struct QueueReactiveBackgroundBundle {
    std::vector<BackgroundHawkesConfig> configs;
    std::vector<int> cluster_ids;
};

// Load the frozen training-only queue-reactive policy mapping.  Every asset
// must have exactly one mapping row.  Cluster policy files determine only the
// shared dynamics; each symbol retains its own stationary event-rate target,
// mark distributions and queue-depth references.
[[nodiscard]] QueueReactiveBackgroundBundle
load_queue_reactive_background_bundle(
    const std::filesystem::path& mapping_csv,
    const std::vector<MultiAssetBookConfig>& assets,
    std::uint64_t simulation_seed,
    int tick_size);

} // namespace dlob
