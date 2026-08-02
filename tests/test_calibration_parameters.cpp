#include "calibration/CalibrationParameters.hpp"

#include <cassert>
#include <cmath>

int main() {
    dlob::calibration::ParameterSpace space;
    dlob::calibration::UnitParameters low{};
    dlob::calibration::UnitParameters high{};
    high.fill(1.0);

    const auto a = space.decode(low);
    const auto b = space.decode(high);

    assert(std::abs(a.market_maker_interval_ms - 5.0) < 1e-12);
    assert(std::abs(b.market_maker_interval_ms - 250.0) < 1e-12);
    assert(a.market_maker_min_spread_ticks == 1);
    assert(b.market_maker_min_spread_ticks == 10);
    assert(std::abs(a.informed_signal_precision - 0.20) < 1e-12);
    assert(std::abs(b.informed_signal_precision - 5.0) < 1e-12);
    assert(std::abs(a.informed_signal_noise_ticks - 5.0) < 1e-12);
    assert(std::abs(b.informed_signal_noise_ticks - 0.2) < 1e-12);
    assert(dlob::calibration::parameter_count == 8);
    assert(dlob::calibration::fixed_hawkes_activity_scale == 0.30);
    return 0;
}
