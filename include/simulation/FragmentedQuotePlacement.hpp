// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#pragma once

#include "common/DistributedTypes.hpp"
#include "simulation/MultiAssetTypes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace dlob::detail {

struct FragmentedQuotePrices {
    std::int64_t bid = 0;
    std::int64_t ask = 0;
};

// Return a direction only when the opposite best quote is executable at a
// sufficiently favourable price.  Midpoint-based direction is deliberately
// excluded because it can be destabilising when the spread is wide.
[[nodiscard]] inline std::optional<Side> fundamental_value_side(
    int best_bid_ticks,
    int best_ask_ticks,
    double fundamental_price_ticks,
    double threshold_bps) noexcept {
    if (best_bid_ticks <= 0 || best_ask_ticks <= best_bid_ticks
        || !(fundamental_price_ticks > 0.0)
        || !std::isfinite(fundamental_price_ticks)
        || !(threshold_bps >= 0.0) || !std::isfinite(threshold_bps)) {
        return std::nullopt;
    }
    const double threshold_fraction = threshold_bps / 10'000.0;
    if (static_cast<double>(best_ask_ticks)
        < fundamental_price_ticks * (1.0 - threshold_fraction)) {
        return Side::Buy;
    }
    if (static_cast<double>(best_bid_ticks)
        > fundamental_price_ticks * (1.0 + threshold_fraction)) {
        return Side::Sell;
    }
    return std::nullopt;
}

// Place contrarian value liquidity without crossing the contemporaneous book.
// When the ask is below value the trader supports the bid; when the bid is
// above value it supplies the ask.  One-tick improvement changes the midpoint
// gradually and leaves execution to subsequent order flow, avoiding the
// mechanically large jumps produced by an aggressive order in a sparse book.
[[nodiscard]] inline int fundamental_value_passive_price(
    Side side,
    int best_bid_ticks,
    int best_ask_ticks,
    double fundamental_price_ticks,
    int tick_size) noexcept {
    if (best_bid_ticks <= 0 || best_ask_ticks <= best_bid_ticks
        || !(fundamental_price_ticks > 0.0)
        || !std::isfinite(fundamental_price_ticks)
        || tick_size <= 0) {
        return 0;
    }
    const std::int64_t tick = tick_size;
    const std::int64_t bid = best_bid_ticks;
    const std::int64_t ask = best_ask_ticks;
    const std::int64_t fundamental_bid = static_cast<std::int64_t>(
        std::floor(fundamental_price_ticks / static_cast<double>(tick))) * tick;
    const std::int64_t fundamental_ask = static_cast<std::int64_t>(
        std::ceil(fundamental_price_ticks / static_cast<double>(tick))) * tick;
    std::int64_t price = 0;
    if (side == Side::Buy) {
        price = std::min({bid + tick, ask - tick, fundamental_bid});
        price = std::max(price, bid);
    } else {
        price = std::max({ask - tick, bid + tick, fundamental_ask});
        price = std::min(price, ask);
    }
    if (price <= 0 || price > std::numeric_limits<std::int32_t>::max()) {
        return 0;
    }
    return static_cast<int>(price);
}

// Magnitude of the executable valuation gap in basis points.  The buy-side
// gap is measured from the best ask up to fundamental value; the sell-side
// gap is measured from fundamental value up to the best bid.  A direction
// that is not currently mispriced has zero gap.
[[nodiscard]] inline double fundamental_value_executable_gap_bps(
    Side side,
    int best_bid_ticks,
    int best_ask_ticks,
    double fundamental_price_ticks) noexcept {
    if (best_bid_ticks <= 0 || best_ask_ticks <= best_bid_ticks
        || !(fundamental_price_ticks > 0.0)
        || !std::isfinite(fundamental_price_ticks)) {
        return 0.0;
    }
    const double executable_price = side == Side::Buy
        ? static_cast<double>(best_ask_ticks)
        : static_cast<double>(best_bid_ticks);
    const double signed_gap = side == Side::Buy
        ? fundamental_price_ticks - executable_price
        : executable_price - fundamental_price_ticks;
    return std::max(
        0.0, 10'000.0 * signed_gap / fundamental_price_ticks);
}

// Gap-sensitive displayed-depth participation:
//
//   p_eff = min(p_max, p_0 * max(g / theta, 1)^eta).
//
// eta=0 is handled before any division or power operation.  This makes the
// default policy exactly, rather than approximately, equivalent to the
// historical fixed-participation rule.  Runtime and CSV validation guarantee
// theta>0 when eta>0 and p_max>=p_0; this helper nevertheless fails closed for
// malformed direct calls.
[[nodiscard]] inline double fundamental_value_effective_participation(
    double base_participation,
    double maximum_participation,
    double executable_gap_bps,
    double threshold_bps,
    double gap_elasticity) noexcept {
    if (!(base_participation > 0.0)
        || !std::isfinite(base_participation)
        || !(maximum_participation > 0.0)
        || !std::isfinite(maximum_participation)
        || base_participation > maximum_participation
        || maximum_participation > 1.0
        || !(gap_elasticity >= 0.0)
        || !std::isfinite(gap_elasticity)) {
        return 0.0;
    }
    if (gap_elasticity == 0.0) {
        return base_participation;
    }
    if (!(threshold_bps > 0.0) || !std::isfinite(threshold_bps)
        || !(executable_gap_bps >= 0.0)
        || !std::isfinite(executable_gap_bps)) {
        return 0.0;
    }
    if (base_participation == maximum_participation) {
        return base_participation;
    }
    const double ratio = std::max(
        executable_gap_bps / threshold_bps, 1.0);
    const double multiplier = std::pow(ratio, gap_elasticity);
    if (!std::isfinite(multiplier)
        || multiplier >= maximum_participation / base_participation) {
        return maximum_participation;
    }
    return std::min(
        maximum_participation, base_participation * multiplier);
}

// Convert a displayed-depth participation rate into a whole-share order size.
// The historical ceil(pD) rule is retained while it describes an executable
// sub-100% participation order.  Its only pathological case is ceil(pD)=D:
// the matching engine can execute only D-1 because the reduced book reserves
// its final displayed share, so a persistent signal repeatedly requests the
// same artificial boundary share.  Capping such an order at D-1 exactly
// matches the quantity the engine would have filled and therefore removes the
// artificial request without changing fills or prices.  An explicit 100%
// policy remains able to exercise the independent finite-boundary diagnostic.
[[nodiscard]] inline int fundamental_value_participation_quantity(
    std::int64_t opposite_depth,
    double participation) noexcept {
    if (opposite_depth <= 0
        || !(participation > 0.0)
        || !std::isfinite(participation)) {
        return 0;
    }
    const std::int64_t bounded_depth = std::min<std::int64_t>(
        opposite_depth, std::numeric_limits<int>::max());
    if (participation >= 1.0) {
        return static_cast<int>(bounded_depth);
    }
    const double desired = participation
        * static_cast<double>(bounded_depth);
    const std::int64_t rounded_up = static_cast<std::int64_t>(
        std::ceil(desired));
    const std::int64_t removable_depth = bounded_depth - 1;
    return static_cast<int>(std::max<std::int64_t>(
        0, std::min(rounded_up, removable_depth)));
}

[[nodiscard]] inline bool deterministic_quote_improvement(
    std::uint64_t model_seed,
    StableEntityId entity,
    BookId book_id,
    std::uint64_t refresh_index,
    double probability) noexcept {
    if (!(probability > 0.0)) return false;
    if (probability >= 1.0) return true;
    constexpr double inverse_two_to_53 = 1.0 / 9'007'199'254'740'992.0;
    const std::uint64_t bits = stable_sequence(
        entity ^ model_seed,
        refresh_index + 1U,
        static_cast<std::uint32_t>(book_id));
    const double draw = static_cast<double>(bits >> 11U) * inverse_two_to_53;
    return draw < probability;
}

// Bounded spread-responsive probability for the sparse local repair maker:
//
//   p_eff = min(p_max, p_0 max(s / s_target, 1)^eta).
//
// eta=0 returns p_0 before any floating-point scaling, preserving the legacy
// deterministic draw bit-for-bit.  Configuration validation supplies finite
// probabilities with 0<=p_0<=p_max<=1 and eta>=0; malformed direct calls fail
// closed to p_0 so this pure helper cannot amplify an invalid input.
[[nodiscard]] inline double local_mm_effective_improvement_probability(
    double base_probability,
    double maximum_probability,
    std::int64_t current_spread_ticks,
    std::int64_t target_spread_ticks,
    double spread_elasticity) noexcept {
    if (spread_elasticity == 0.0) {
        return base_probability;
    }
    if (!(base_probability >= 0.0) || !std::isfinite(base_probability)
        || !(maximum_probability >= base_probability)
        || maximum_probability > 1.0
        || !std::isfinite(maximum_probability)
        || !(spread_elasticity > 0.0) || !std::isfinite(spread_elasticity)
        || current_spread_ticks <= 0 || target_spread_ticks <= 0) {
        return base_probability;
    }
    if (base_probability == 0.0
        || base_probability == maximum_probability
        || current_spread_ticks <= target_spread_ticks) {
        return base_probability;
    }
    const double ratio = std::max(
        static_cast<double>(current_spread_ticks)
            / static_cast<double>(target_spread_ticks),
        1.0);
    const double multiplier = std::pow(ratio, spread_elasticity);
    if (!std::isfinite(multiplier)
        || multiplier >= maximum_probability / base_probability) {
        return maximum_probability;
    }
    return std::min(maximum_probability, base_probability * multiplier);
}

// The local maker is a sparse structural repair mechanism, not a continuous
// synthetic depth floor.  In reactive mode it restores a missing side,
// replenishes a top queue that has fallen below its calibrated quote size, and
// may tighten a two-sided book only when the spread is wider than the empirical
// target and the deterministic improvement draw succeeds.
// Keeping this decision in a pure helper makes the model semantics directly
// testable and prevents a Boolean call-site regression from restoring the
// over-damping observed in development-validation job 45271.
[[nodiscard]] inline bool fragmented_quote_required(
    bool quote_only_when_repairing,
    bool one_sided,
    bool shallow_top,
    bool wide_spread,
    bool improve_wide_spread) noexcept {
    return !quote_only_when_repairing
        || one_sided
        || shallow_top
        || (wide_spread && improve_wide_spread);
}

// Determine one maker's quote prices from the contemporaneous local book.
// Missing-side repair preserves the side that still exists; the fixed
// opening fundamental is used only if both sides are absent.  When the book
// is two-sided, an inside-spread update is optional rather than a hard spread
// ceiling.
[[nodiscard]] inline FragmentedQuotePrices fragmented_quote_prices(
    const MarketState& state,
    double fundamental_price_ticks,
    int tick_size,
    int target_spread_ticks,
    bool improve_wide_spread) {
    const std::int64_t tick = std::max<std::int64_t>(1, tick_size);
    const std::int64_t spread_ticks = std::max<std::int64_t>(
        1, target_spread_ticks);
    const std::int64_t target_spread = tick * spread_ticks;
    std::int64_t bid = state.best_bid_ticks;
    std::int64_t ask = state.best_ask_ticks;

    if (bid > 0 && ask <= 0) {
        // Preserve the surviving bid and reconstruct only the missing ask.
        ask = bid + target_spread;
    } else if (ask > 0 && bid <= 0) {
        // Preserve the surviving ask and reconstruct only the missing bid.
        bid = ask - target_spread;
    } else if (bid <= 0 && ask <= 0) {
        const auto center = static_cast<std::int64_t>(std::llround(
            fundamental_price_ticks / static_cast<double>(tick))) * tick;
        bid = center - (spread_ticks / 2) * tick;
        ask = bid + target_spread;
    } else if (ask <= bid) {
        // A crossed stored book should be unreachable because matching occurs
        // immediately.  If encountered, preserve the bid and reconstruct the
        // ask locally rather than jumping to the fixed opening reference.
        ask = bid + target_spread;
    } else if (ask - bid > target_spread && improve_wide_spread) {
        // One refresh improves each eligible side by at most one tick.  The
        // former implementation jumped directly to the target spread around
        // a possibly stale last trade, creating rare quote-driven price
        // discontinuities.  Tighten gradually and never overshoot the target.
        if (ask - (bid + tick) >= target_spread) {
            bid += tick;
        }
        if ((ask - tick) - bid >= target_spread) {
            ask -= tick;
        }
    }
    return {bid, ask};
}

} // namespace dlob::detail
