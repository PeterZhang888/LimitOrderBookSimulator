// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#pragma once

#include "agents/SharedMarketMakerAgent.hpp"
#include "exchange/BackgroundHawkesAgent.hpp"
#include "simulation/MultiAssetTypes.hpp"

#include <filesystem>
#include <vector>

namespace dlob {

[[nodiscard]] std::vector<MultiAssetBookConfig> load_multi_asset_book_configs(
    const std::filesystem::path& path);

[[nodiscard]] std::vector<MultiAssetBookConfig> resolve_multi_asset_book_configs(
    const SequentialMultiAssetConfig& config);

[[nodiscard]] BackgroundHawkesConfig make_multi_asset_background_config(
    const SequentialMultiAssetConfig& config,
    const MultiAssetBookConfig& book,
    BookId book_id);

[[nodiscard]] SharedMarketMakerConfig make_multi_asset_market_maker_config(
    const SequentialMultiAssetConfig& config,
    const std::vector<MultiAssetBookConfig>& books);

} // namespace dlob
