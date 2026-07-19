#include "common/EmpiricalDistribution.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>

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
} // namespace

void EmpiricalDistribution::set_fallback(int lower, int upper) {
    fallback_lower_ = std::max(0, lower);
    fallback_upper_ = std::max(fallback_lower_, upper);
}

bool EmpiricalDistribution::load_from_csv(const std::string& filename, const std::string& column_name) {
    values_.clear();
    std::ifstream input(filename);
    if (!input.is_open()) return false;

    std::string header;
    if (!std::getline(input, header)) return false;
    const std::vector<std::string> headers = split_csv_line(header);
    int column = -1;
    for (std::size_t i = 0; i < headers.size(); ++i) {
        if (headers[i] == column_name) {
            column = static_cast<int>(i);
            break;
        }
    }
    if (column < 0) column = 0;

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> fields = split_csv_line(line);
        if (column >= static_cast<int>(fields.size())) continue;
        try {
            const int value = static_cast<int>(std::llround(std::stod(fields[static_cast<std::size_t>(column)])));
            if (value >= fallback_lower_) values_.push_back(value);
        } catch (...) {
            continue;
        }
    }
    return !values_.empty();
}

int EmpiricalDistribution::sample(FastRng& rng) const {
    if (!values_.empty()) {
        return std::max(fallback_lower_, values_[static_cast<std::size_t>(rng.uniform_int(0, static_cast<int>(values_.size()) - 1))]);
    }
    return std::max(fallback_lower_, rng.uniform_int(fallback_lower_, fallback_upper_));
}

} // namespace dlob
