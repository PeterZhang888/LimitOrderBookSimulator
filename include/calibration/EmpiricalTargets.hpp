#pragma once

#include "calibration/SimulationRecorder.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace dlob::calibration {

struct WeightedDistribution {
    std::vector<int> support;
    std::vector<double> cumulative_probability;
    double total_count = 0.0;

    static WeightedDistribution load(const std::filesystem::path& path,
                                     const std::string& value_column = "quantity");
    double cdf(int value) const;
    double quantile(double probability) const;
};

struct MarketTarget {
    std::string name;
    double target = 0.0;
    double scale = 1.0;
    double weight = 1.0;
};

struct DistanceBreakdown {
    double total = 0.0;
    std::array<double, empirical_event_bucket_count> quantity_ks{};
    double event_proportion_l1 = 0.0;
    double market_component = 0.0;
};

class EmpiricalTargets {
public:
    static EmpiricalTargets load(const std::filesystem::path& data_directory,
                                 const std::filesystem::path& market_target_csv = {});

    DistanceBreakdown distance(const SimulationRecord& record) const;
    const std::array<WeightedDistribution, empirical_event_bucket_count>& distributions() const {
        return distributions_;
    }
    const std::vector<MarketTarget>& market_targets() const { return market_targets_; }

private:
    static double ks_distance(const WeightedDistribution& target,
                              std::vector<int> sample);
    static double market_value(const MarketFeatureSummary& summary,
                               const std::string& name);

    std::array<WeightedDistribution, empirical_event_bucket_count> distributions_{};
    std::array<double, empirical_event_bucket_count> event_proportions_{};
    std::vector<MarketTarget> market_targets_;
};

} // namespace dlob::calibration
