#pragma once

#include "MomentEstimator.hpp"

#include <string>
#include <vector>

struct LossTarget {
    std::string name;
    double target = 0.0;
    double weight = 1.0;
    double scale_floor = 1e-6;
};

struct LossBreakdown {
    double total_loss = 0.0;
    std::vector<NamedMoment> simulated_moments;
    std::vector<NamedMoment> loss_terms;
};

std::vector<LossTarget> load_loss_targets(const std::string& filename);
LossBreakdown compute_loss(const std::vector<NamedMoment>& simulated, const std::vector<LossTarget>& targets);
