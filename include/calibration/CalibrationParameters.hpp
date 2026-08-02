// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace dlob::calibration {

inline constexpr std::size_t parameter_count = 8;
inline constexpr double fixed_hawkes_activity_scale = 0.30;

using UnitParameters = std::array<double, parameter_count>;

enum class ParameterKind { Continuous, Integer };

struct ParameterSpec {
    std::string name;
    double lower = 0.0;
    double upper = 1.0;
    ParameterKind kind = ParameterKind::Continuous;
};

// The population composition and empirical order-size distributions are fixed.
// Calibration focuses on two economically interpretable capabilities per
// strategic agent class.
struct PhysicalParameters {
    int market_makers = 3;
    int momentum_traders = 6'000;
    int informed_traders = 2'900;
    int institutional_traders = 100;

    int market_maker_order_quantity = 100;
    int momentum_order_quantity = 100;
    int informed_base_quantity = 100;
    double market_maker_quote_skip_probability = 0.05;
    double informed_trade_threshold_ticks = 1.0;

    // 1-2: liquidity provision capability.
    double market_maker_interval_ms = 20.0;
    int market_maker_min_spread_ticks = 2;

    // 3-4: momentum capability.
    double momentum_rate_per_second = 0.20;
    double momentum_threshold_ticks = 0.25;

    // 5-6: informed-trading capability.
    double informed_rate_per_second = 0.05;
    double informed_signal_precision = 1.0;
    double informed_signal_noise_ticks = 1.0;

    // 7-8: institutional execution capability.
    double institutional_rate_per_second = 0.01;
    double institutional_participation_cap = 0.10;
};

class ParameterSpace {
public:
    ParameterSpace();
    explicit ParameterSpace(std::vector<ParameterSpec> specs);

    static ParameterSpace load_csv(const std::filesystem::path& path);
    static void write_default_csv(const std::filesystem::path& path);

    const std::array<ParameterSpec, parameter_count>& specs() const noexcept { return specs_; }
    PhysicalParameters decode(const UnitParameters& unit) const;
    UnitParameters clamp(const UnitParameters& unit) const;

private:
    std::array<ParameterSpec, parameter_count> specs_{};
};

const std::array<const char*, parameter_count>& parameter_names();

} // namespace dlob::calibration
