#include "EmpiricalDistribution.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

EmpiricalDistribution::EmpiricalDistribution() {}

std::size_t EmpiricalDistribution::size() const {
    return values_.size();
}

void EmpiricalDistribution::load_from_csv(
    const std::string& file_path,
    const std::string& value_column
) {
    values_.clear();
    cumulative_probabilities_.clear();

    std::ifstream file(file_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "EmpiricalDistribution: cannot open file: " + file_path
        );
    }

    // =========================
    // 1. Read header
    // =========================

    std::string header_line;

    if (!std::getline(file, header_line)) {
        throw std::runtime_error(
            "EmpiricalDistribution: empty file: " + file_path
        );
    }

    std::vector<std::string> headers;
    std::stringstream header_stream(header_line);
    std::string cell;

    while (std::getline(header_stream, cell, ',')) {
        headers.push_back(cell);
    }

    int value_col_index = -1;
    int cumulative_col_index = -1;

    for (int i = 0; i < static_cast<int>(headers.size()); ++i) {
        if (headers[i] == value_column) {
            value_col_index = i;
        }

        if (headers[i] == "cumulative_probability") {
            cumulative_col_index = i;
        }
    }

    if (value_col_index == -1) {
        throw std::runtime_error(
            "EmpiricalDistribution: value column not found: " + value_column
        );
    }

    if (cumulative_col_index == -1) {
        throw std::runtime_error(
            "EmpiricalDistribution: cumulative_probability column not found in: "
            + file_path
        );
    }

    // =========================
    // 2. Read rows
    // =========================

    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> fields;
        std::stringstream line_stream(line);
        std::string field;

        while (std::getline(line_stream, field, ',')) {
            fields.push_back(field);
        }

        if (
            value_col_index >= static_cast<int>(fields.size()) ||
            cumulative_col_index >= static_cast<int>(fields.size())
        ) {
            continue;
        }

        int value = std::stoi(fields[value_col_index]);
        double cumulative_probability =
            std::stod(fields[cumulative_col_index]);

        values_.push_back(value);
        cumulative_probabilities_.push_back(cumulative_probability);
    }

    if (values_.empty()) {
        throw std::runtime_error(
            "EmpiricalDistribution: no data loaded from file: " + file_path
        );
    }

    // =========================
    // 3. Basic safety checks
    // =========================

    for (std::size_t i = 1; i < cumulative_probabilities_.size(); ++i) {
        if (cumulative_probabilities_[i] < cumulative_probabilities_[i - 1]) {
            throw std::runtime_error(
                "EmpiricalDistribution: cumulative_probability is not increasing in: "
                + file_path
            );
        }
    }

    // Force last cumulative probability to 1.0 to avoid rounding issues.
    cumulative_probabilities_.back() = 1.0;
}

int EmpiricalDistribution::sample(std::mt19937_64& rng) const {
    if (values_.empty()) {
        throw std::runtime_error(
            "EmpiricalDistribution: cannot sample from empty distribution."
        );
    }

    // =========================
    // Your method:
    // 1. Generate u ~ Uniform(0, 1)
    // 2. Find first cumulative_probability >= u
    // 3. Return corresponding value
    // =========================

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double u = uniform(rng);

    auto it = std::lower_bound(
        cumulative_probabilities_.begin(),
        cumulative_probabilities_.end(),
        u
    );

    if (it == cumulative_probabilities_.end()) {
        return values_.back();
    }

    std::size_t index = static_cast<std::size_t>(
        std::distance(cumulative_probabilities_.begin(), it)
    );

    return values_[index];
}
