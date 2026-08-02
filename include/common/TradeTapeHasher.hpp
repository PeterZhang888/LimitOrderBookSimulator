#pragma once

#include "common/DistributedTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace dlob {

// Deterministic FNV-1a over explicitly serialized fields.  Raw struct bytes are
// deliberately never hashed: their padding and host byte order are not stable
// across compilers or machines.  Each integer below is serialized big-endian.
class TradeTapeHasher {
public:
    static constexpr std::uint64_t offset_basis = 14695981039346656037ULL;
    static constexpr std::uint64_t prime = 1099511628211ULL;

    void add(const TradeExecution& trade) noexcept {
        append_integral(trade.book_id);
        append_integral(trade.timestamp_ns);
        append_integral(trade.trade_sequence);
        append_integral(trade.price_ticks);
        append_integral(trade.quantity);
        append_integral(trade.buyer_owner_id);
        append_integral(trade.seller_owner_id);
        append_integral(trade.buyer_order_sequence);
        append_integral(trade.seller_order_sequence);
        append_integral(static_cast<std::int32_t>(trade.aggressor_side));
        append_integral(static_cast<std::int32_t>(trade.aggressor_action));
        ++trade_count_;
    }

    void update(const TradeExecution& trade) noexcept { add(trade); }

    [[nodiscard]] std::uint64_t digest() const noexcept { return hash_; }
    [[nodiscard]] std::uint64_t value() const noexcept { return digest(); }
    [[nodiscard]] std::uint64_t trade_count() const noexcept { return trade_count_; }

    [[nodiscard]] static std::uint64_t hash(std::span<const TradeExecution> trades) noexcept {
        TradeTapeHasher hasher;
        for (const TradeExecution& trade : trades) hasher.add(trade);
        return hasher.digest();
    }

private:
    template <typename Integer>
    void append_integral(Integer value) noexcept {
        static_assert(std::is_integral_v<Integer>);
        using Unsigned = std::make_unsigned_t<Integer>;
        const Unsigned bits = static_cast<Unsigned>(value);
        for (std::size_t index = 0; index < sizeof(Unsigned); ++index) {
            const std::size_t shift = (sizeof(Unsigned) - index - 1U) * 8U;
            const auto byte = static_cast<std::uint8_t>(bits >> shift);
            hash_ ^= byte;
            hash_ *= prime;
        }
    }

    std::uint64_t hash_ = offset_basis;
    std::uint64_t trade_count_ = 0;
};

[[nodiscard]] inline std::uint64_t
hash_trade_tape(std::span<const TradeExecution> trades) noexcept {
    return TradeTapeHasher::hash(trades);
}

} // namespace dlob
