#include "calibration/EmpiricalTargets.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace dlob::calibration {
namespace {

std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    std::size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) ++first;
    value.erase(0, first);
    return value;
}

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string cell;
    while (std::getline(stream, cell, ',')) fields.push_back(trim(cell));
    return fields;
}

int find_column(const std::vector<std::string>& header,
                std::initializer_list<std::string_view> names) {
    for (std::string_view name : names) {
        for (std::size_t i = 0; i < header.size(); ++i) {
            if (header[i] == name) return static_cast<int>(i);
        }
    }
    return -1;
}

const std::array<const char*, empirical_event_bucket_count> filenames{
    "limit_buy_quantity_distribution.txt",
    "limit_sell_quantity_distribution.txt",
    "market_buy_quantity_distribution.txt",
    "market_sell_quantity_distribution.txt",
    "cancel_bid_quantity_distribution.txt",
    "cancel_ask_quantity_distribution.txt"
};

} // namespace

WeightedDistribution WeightedDistribution::load(const std::filesystem::path& path,
                                                const std::string& value_column_name) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open empirical distribution: " + path.string());
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Empty empirical distribution: " + path.string());
    const auto header = split(line);
    const int value_column = find_column(header, {value_column_name, "quantity", "size"});
    const int count_column = find_column(header, {"count", "frequency", "weight"});
    const int probability_column = find_column(header, {"probability", "prob", "mass"});
    if (value_column < 0) throw std::runtime_error("No quantity column in: " + path.string());

    std::vector<std::pair<int, double>> rows;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        const auto fields = split(line);
        if (value_column >= static_cast<int>(fields.size())) continue;
        try {
            const int value = static_cast<int>(std::llround(std::stod(fields[static_cast<std::size_t>(value_column)])));
            double weight = 1.0;
            if (count_column >= 0 && count_column < static_cast<int>(fields.size())
                && !fields[static_cast<std::size_t>(count_column)].empty()) {
                weight = std::stod(fields[static_cast<std::size_t>(count_column)]);
            } else if (probability_column >= 0 && probability_column < static_cast<int>(fields.size())
                       && !fields[static_cast<std::size_t>(probability_column)].empty()) {
                weight = std::stod(fields[static_cast<std::size_t>(probability_column)]);
            }
            if (std::isfinite(weight) && weight > 0.0 && value >= 0) rows.emplace_back(value, weight);
        } catch (...) {
        }
    }
    if (rows.empty()) throw std::runtime_error("No valid rows in empirical distribution: " + path.string());
    std::sort(rows.begin(), rows.end());

    WeightedDistribution out;
    std::vector<double> weights;
    for (const auto& [value, weight] : rows) {
        if (!out.support.empty() && out.support.back() == value) weights.back() += weight;
        else {
            out.support.push_back(value);
            weights.push_back(weight);
        }
    }
    out.total_count = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (!(out.total_count > 0.0)) throw std::runtime_error("Non-positive distribution mass: " + path.string());
    out.cumulative_probability.resize(weights.size());
    double cumulative = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i] / out.total_count;
        out.cumulative_probability[i] = cumulative;
    }
    out.cumulative_probability.back() = 1.0;
    return out;
}

double WeightedDistribution::cdf(int value) const {
    const auto it = std::upper_bound(support.begin(), support.end(), value);
    if (it == support.begin()) return 0.0;
    const std::size_t index = static_cast<std::size_t>(std::distance(support.begin(), it) - 1);
    return cumulative_probability[index];
}

double WeightedDistribution::quantile(double probability) const {
    if (support.empty()) return 0.0;
    const double p = std::clamp(probability, 0.0, 1.0);
    const auto it = std::lower_bound(cumulative_probability.begin(), cumulative_probability.end(), p);
    const std::size_t index = it == cumulative_probability.end()
        ? cumulative_probability.size() - 1
        : static_cast<std::size_t>(std::distance(cumulative_probability.begin(), it));
    return static_cast<double>(support[index]);
}

EmpiricalTargets EmpiricalTargets::load(const std::filesystem::path& data_directory,
                                        const std::filesystem::path& market_target_csv) {
    EmpiricalTargets targets;
    double total_count = 0.0;
    for (std::size_t i = 0; i < empirical_event_bucket_count; ++i) {
        std::filesystem::path path = data_directory / filenames[i];
        if (!std::filesystem::exists(path)) path = filenames[i];
        targets.distributions_[i] = WeightedDistribution::load(path);
        total_count += targets.distributions_[i].total_count;
    }
    for (std::size_t i = 0; i < empirical_event_bucket_count; ++i) {
        targets.event_proportions_[i] = targets.distributions_[i].total_count / total_count;
    }

    if (!market_target_csv.empty() && std::filesystem::exists(market_target_csv)) {
        std::ifstream input(market_target_csv);
        std::string line;
        if (!std::getline(input, line)) throw std::runtime_error("Empty market target CSV");
        const auto header = split(line);
        const int name_column = find_column(header, {"name", "metric", "moment_name"});
        const int target_column = find_column(header, {"target", "value", "empirical_value"});
        const int scale_column = find_column(header, {"scale", "std", "scale_floor"});
        const int weight_column = find_column(header, {"weight", "loss_weight"});
        if (name_column < 0 || target_column < 0) {
            throw std::runtime_error("Market target CSV requires name,target columns");
        }
        while (std::getline(input, line)) {
            if (line.empty() || line.front() == '#') continue;
            const auto fields = split(line);
            if (name_column >= static_cast<int>(fields.size())
                || target_column >= static_cast<int>(fields.size())) continue;
            MarketTarget target;
            target.name = fields[static_cast<std::size_t>(name_column)];
            target.target = std::stod(fields[static_cast<std::size_t>(target_column)]);
            if (scale_column >= 0 && scale_column < static_cast<int>(fields.size())
                && !fields[static_cast<std::size_t>(scale_column)].empty()) {
                target.scale = std::max(1e-12, std::abs(std::stod(fields[static_cast<std::size_t>(scale_column)])));
            } else {
                target.scale = std::max(1e-8, 0.10 * std::abs(target.target));
            }
            if (weight_column >= 0 && weight_column < static_cast<int>(fields.size())
                && !fields[static_cast<std::size_t>(weight_column)].empty()) {
                target.weight = std::max(0.0, std::stod(fields[static_cast<std::size_t>(weight_column)]));
            }
            targets.market_targets_.push_back(std::move(target));
        }
    }
    return targets;
}

double EmpiricalTargets::ks_distance(const WeightedDistribution& target,
                                     std::vector<int> sample) {
    if (sample.empty()) return 1.0;
    std::sort(sample.begin(), sample.end());
    double maximum = 0.0;
    std::size_t sample_index = 0;
    std::size_t target_index = 0;
    while (sample_index < sample.size() || target_index < target.support.size()) {
        int value = 0;
        if (sample_index >= sample.size()) value = target.support[target_index];
        else if (target_index >= target.support.size()) value = sample[sample_index];
        else value = std::min(sample[sample_index], target.support[target_index]);

        while (sample_index < sample.size() && sample[sample_index] <= value) ++sample_index;
        while (target_index < target.support.size() && target.support[target_index] <= value) ++target_index;

        const double sample_cdf = static_cast<double>(sample_index) / static_cast<double>(sample.size());
        const double target_cdf = target.cdf(value);
        maximum = std::max(maximum, std::abs(sample_cdf - target_cdf));
    }
    return maximum;
}

double EmpiricalTargets::market_value(const MarketFeatureSummary& summary,
                                      const std::string& name) {
    if (name == "mean_spread_ticks") return summary.mean_spread_ticks;
    if (name == "mean_bid_depth") return summary.mean_bid_depth;
    if (name == "mean_ask_depth") return summary.mean_ask_depth;
    if (name == "mid_move_rate") return summary.mid_move_rate;
    if (name == "return_variance") return summary.return_variance;
    if (name == "return_kurtosis") return summary.return_kurtosis;
    if (name == "absolute_return_acf1") return summary.absolute_return_acf1;
    throw std::runtime_error("Unknown market target name: " + name);
}

DistanceBreakdown EmpiricalTargets::distance(const SimulationRecord& record) const {
    DistanceBreakdown breakdown;
    double ks_sum2 = 0.0;
    std::uint64_t total_events = 0;
    for (std::uint64_t count : record.event_counts) total_events += count;

    for (std::size_t i = 0; i < empirical_event_bucket_count; ++i) {
        breakdown.quantity_ks[i] = ks_distance(distributions_[i], record.quantity_samples[i]);
        ks_sum2 += breakdown.quantity_ks[i] * breakdown.quantity_ks[i];
        const double simulated = total_events > 0
            ? static_cast<double>(record.event_counts[i]) / static_cast<double>(total_events)
            : 0.0;
        breakdown.event_proportion_l1 += std::abs(simulated - event_proportions_[i]);
    }

    double market_weight_sum = 0.0;
    double market_sum = 0.0;
    for (const MarketTarget& target : market_targets_) {
        const double residual = (market_value(record.market, target.name) - target.target) / target.scale;
        market_sum += target.weight * residual * residual;
        market_weight_sum += target.weight;
    }
    breakdown.market_component = market_weight_sum > 0.0
        ? std::sqrt(market_sum / market_weight_sum)
        : 0.0;

    const double ks_component = std::sqrt(ks_sum2 / static_cast<double>(empirical_event_bucket_count));
    const double event_component = breakdown.event_proportion_l1 / 2.0;
    // Quantity-distribution fit is mandatory. Event composition and optional
    // market moments are additional terms on comparable scales.
    breakdown.total = std::sqrt(
        ks_component * ks_component
        + 0.50 * event_component * event_component
        + breakdown.market_component * breakdown.market_component);
    if (!std::isfinite(breakdown.total)) breakdown.total = 1e9;
    return breakdown;
}

} // namespace dlob::calibration
