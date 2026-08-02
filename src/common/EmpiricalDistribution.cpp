// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#include "common/EmpiricalDistribution.hpp"

#include "common/DataPaths.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string_view>

namespace dlob {
namespace {

std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) ++start;
    if (start > 0) value.erase(0, start);
    return value;
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string cell;
    while (std::getline(stream, cell, ',')) fields.push_back(trim(cell));
    return fields;
}

int find_column(const std::vector<std::string>& header,
                std::initializer_list<std::string_view> candidates) {
    for (std::string_view candidate : candidates) {
        for (std::size_t i = 0; i < header.size(); ++i) {
            if (header[i] == candidate) return static_cast<int>(i);
        }
    }
    return -1;
}

} // namespace

void EmpiricalDistribution::set_fallback(int lower, int upper) {
    fallback_lower_ = std::max(0, lower);
    fallback_upper_ = std::max(fallback_lower_, upper);
}

bool EmpiricalDistribution::load_from_csv(const std::string& filename,
                                          const std::string& column_name) {
    values_.clear();
    cumulative_weights_.clear();

    std::ifstream input(resolve_data_file(filename));
    if (!input.is_open()) return false;

    std::string header_line;
    if (!std::getline(input, header_line)) return false;
    const std::vector<std::string> header = split_csv_line(header_line);

    int value_column = find_column(header, {column_name});
    if (value_column < 0 && column_name == "distance_ticks") {
        value_column = find_column(header, {"distance", "distance_ticks"});
    }
    if (value_column < 0 && column_name == "quantity") {
        value_column = find_column(header, {"quantity", "size"});
    }
    if (value_column < 0) return false;

    const int count_column = find_column(header, {"count", "frequency", "weight"});
    const int probability_column = find_column(header, {"probability", "prob", "mass"});

    std::vector<double> weights;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        const std::vector<std::string> fields = split_csv_line(line);
        if (value_column >= static_cast<int>(fields.size())) continue;
        try {
            const double raw = std::stod(fields[static_cast<std::size_t>(value_column)]);
            if (!std::isfinite(raw)) continue;
            const int value = static_cast<int>(std::llround(raw));
            if (value < fallback_lower_) continue;

            double weight = 1.0;
            if (count_column >= 0 && count_column < static_cast<int>(fields.size())
                && !fields[static_cast<std::size_t>(count_column)].empty()) {
                weight = std::stod(fields[static_cast<std::size_t>(count_column)]);
            } else if (probability_column >= 0
                       && probability_column < static_cast<int>(fields.size())
                       && !fields[static_cast<std::size_t>(probability_column)].empty()) {
                weight = std::stod(fields[static_cast<std::size_t>(probability_column)]);
            }
            if (!std::isfinite(weight) || weight <= 0.0) continue;
            values_.push_back(value);
            weights.push_back(weight);
        } catch (...) {
            continue;
        }
    }

    if (values_.empty()) return false;

    std::vector<std::size_t> order(values_.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [this](std::size_t a, std::size_t b) {
        return values_[a] < values_[b];
    });

    std::vector<int> sorted_values;
    std::vector<double> sorted_weights;
    sorted_values.reserve(values_.size());
    sorted_weights.reserve(weights.size());
    for (std::size_t index : order) {
        if (!sorted_values.empty() && sorted_values.back() == values_[index]) {
            sorted_weights.back() += weights[index];
        } else {
            sorted_values.push_back(values_[index]);
            sorted_weights.push_back(weights[index]);
        }
    }
    values_.swap(sorted_values);

    const double total = std::accumulate(sorted_weights.begin(), sorted_weights.end(), 0.0);
    if (!(total > 0.0) || !std::isfinite(total)) {
        values_.clear();
        return false;
    }

    cumulative_weights_.resize(sorted_weights.size());
    double cumulative = 0.0;
    for (std::size_t i = 0; i < sorted_weights.size(); ++i) {
        cumulative += sorted_weights[i] / total;
        cumulative_weights_[i] = cumulative;
    }
    cumulative_weights_.back() = 1.0;
    return true;
}

int EmpiricalDistribution::sample(FastRng& rng) const {
    if (!values_.empty()) {
        const double draw = rng.uniform01();
        const auto it = std::lower_bound(cumulative_weights_.begin(), cumulative_weights_.end(), draw);
        const std::size_t index = it == cumulative_weights_.end()
            ? cumulative_weights_.size() - 1
            : static_cast<std::size_t>(std::distance(cumulative_weights_.begin(), it));
        return std::max(fallback_lower_, values_[index]);
    }
    return std::max(fallback_lower_, rng.uniform_int(fallback_lower_, fallback_upper_));
}

double EmpiricalDistribution::probability_mass(int value) const noexcept {
    const auto found = std::lower_bound(values_.begin(), values_.end(), value);
    if (found == values_.end() || *found != value) return 0.0;
    const std::size_t index = static_cast<std::size_t>(
        std::distance(values_.begin(), found));
    const double lower = index == 0 ? 0.0 : cumulative_weights_[index - 1U];
    return std::max(0.0, cumulative_weights_[index] - lower);
}

} // namespace dlob
