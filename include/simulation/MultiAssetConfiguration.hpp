#pragma once

#include "exchange/BackgroundHawkesAgent.hpp"
#include "simulation/MultiAssetTypes.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace dlob {

[[nodiscard]] std::vector<MultiAssetBookConfig> load_multi_asset_book_configs(
    const std::filesystem::path& path);

[[nodiscard]] BackgroundHawkesConfig make_multi_asset_background_config(
    const MultiAssetBookConfig& book,
    BookId book_id,
    std::uint64_t seed,
    int tick_size);

} // namespace dlob
