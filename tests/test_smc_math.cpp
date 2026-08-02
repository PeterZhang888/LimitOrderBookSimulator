#include "calibration/SmcAbc.hpp"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
    using namespace dlob::calibration;
    const double q = weighted_quantile({1.0, 2.0, 3.0}, {0.2, 0.3, 0.5}, 0.5);
    assert(q == 2.0 || q == 3.0);

    std::vector<Particle> particles(2);
    particles[0].weight = 0.5;
    particles[1].weight = 0.5;
    particles[0].theta.fill(0.25);
    particles[1].theta.fill(0.75);
    const auto variance = diagonal_kernel_variance(particles);
    for (double v : variance) assert(std::abs(v - 0.125) < 1e-12);

    const auto weights = normalize_log_weights({0.0, 0.0});
    assert(weights.size() == 2);
    assert(std::abs(weights[0] - 0.5) < 1e-12);
    assert(std::abs(effective_sample_size(weights) - 2.0) < 1e-12);
    return 0;
}
