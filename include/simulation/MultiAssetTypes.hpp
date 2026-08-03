#pragma once

#include "common/DistributedTypes.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace dlob {

// Logical entity identifiers are independent of MPI ownership.  Repartitioning
// books therefore cannot change random streams or deterministic tie-breaks.
using StableEntityId = std::uint64_t;

[[nodiscard]] inline StableEntityId stable_symbol_stream_id(
    std::string_view symbol) noexcept {
    StableEntityId hash = 14'695'981'039'346'656'037ULL;
    for (const char character : symbol) {
        const auto byte = static_cast<unsigned char>(character);
        hash ^= static_cast<StableEntityId>(byte);
        hash *= 1'099'511'628'211ULL;
    }
    return hash ^ 0xa076'1d64'78bd'642fULL;
}

inline constexpr StableEntityId background_entity_base = 0x0001'0000ULL;
inline constexpr StableEntityId liquidity_shock_entity_base = 0x0005'0000ULL;
inline constexpr std::int32_t liquidity_shock_owner_id = 800'001;
inline constexpr std::int32_t fundamental_value_owner_id_base = 600'001;
inline constexpr std::int32_t local_market_maker_owner_id_base = 500'001;

[[nodiscard]] constexpr StableEntityId background_entity(
    BookId book_id) noexcept {
    return background_entity_base + static_cast<StableEntityId>(book_id);
}

[[nodiscard]] constexpr StableEntityId liquidity_shock_entity(
    BookId book_id) noexcept {
    return liquidity_shock_entity_base + static_cast<StableEntityId>(book_id);
}

[[nodiscard]] constexpr std::int32_t fundamental_value_owner_id(
    BookId book_id) noexcept {
    return fundamental_value_owner_id_base + static_cast<std::int32_t>(book_id);
}

[[nodiscard]] constexpr std::int32_t local_market_maker_owner_id(
    BookId book_id) noexcept {
    return local_market_maker_owner_id_base + static_cast<std::int32_t>(book_id);
}

// Empirical configuration for one logical book.  The final simulator uses one
// such row per symbol; it does not replicate a synthetic template.
struct MultiAssetBookConfig {
    std::string symbol;
    std::string data_dir;
    std::string hawkes_rates_file;
    double fundamental_price_ticks = 0.0;
    double fundamental_volatility_bps_sqrt_second = 0.0;
    double fundamental_move_probability_per_second = 1.0;
    double fundamental_conditional_kurtosis = 3.0;
    double fundamental_log_variance_persistence = 0.0;
    double fundamental_log_variance_std = 0.0;
    double fundamental_order_flow_coupling = 0.0;
    std::int32_t initial_best_bid_ticks = 0;
    std::int32_t initial_best_ask_ticks = 0;
    std::int32_t initial_best_bid_depth = 0;
    std::int32_t initial_best_ask_depth = 0;
    double target_mean_bid_depth = 0.0;
    double target_mean_ask_depth = 0.0;
    double beta = 1.0;
    double basket_weight = 0.0;
    std::int32_t market_maker_quote_quantity = 0;
    std::int32_t target_spread_ticks = 1;
    double quote_improvement_probability = 0.05;
};

[[nodiscard]] constexpr std::uint64_t stable_sequence(
    StableEntityId entity,
    std::uint64_t local_sequence,
    std::uint32_t child_index = 0) noexcept {
    std::uint64_t value = entity;
    value ^= local_sequence + 0x9e3779b97f4a7c15ULL
        + (value << 6U) + (value >> 2U);
    value ^= static_cast<std::uint64_t>(child_index)
        * 0xd1b54a32d192ed03ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

} // namespace dlob
