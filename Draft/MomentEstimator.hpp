#pragma once

#include "SimulationMetrics.hpp"

#include <string>
#include <vector>

struct NamedMoment {
    std::string name;
    double value = 0.0;
};

const std::vector<std::string>& standard_moment_names();
std::vector<NamedMoment> estimate_moments(const SimulationMetrics& metrics);
double get_moment_value(const std::vector<NamedMoment>& moments, const std::string& name, double fallback = 0.0);
