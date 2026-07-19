#include "EmpiricalDistribution.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

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
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) fields.push_back(trim(cell));
    return fields;
}
}

void EmpiricalDistribution::set_fallback(int lower, int upper) {
    fallback_lower_ = std::max(1, lower);
    fallback_upper_ = std::max(fallback_lower_, upper);
}

bool EmpiricalDistribution::load_from_csv(const std::string& filename, const std::string& column_name) {
    values_.clear();
    std::ifstream input(filename);
    if (!input.is_open()) return false;

    std::string header;
    if (!std::getline(input, header)) return false;
    std::vector<std::string> headers = split_csv_line(header);
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
        std::vector<std::string> fields = split_csv_line(line);
        if (column >= static_cast<int>(fields.size())) continue;
        try {
            const int value = static_cast<int>(std::llround(std::stod(fields[column])));
            if (value > 0) values_.push_back(value);
        } catch (...) {
            continue;
        }
    }
    return !values_.empty();
}

int EmpiricalDistribution::sample(std::mt19937_64& rng) const {
    if (!values_.empty()) {
        std::uniform_int_distribution<std::size_t> dist(0, values_.size() - 1);
        return std::max(1, values_[dist(rng)]);
    }
    std::uniform_int_distribution<int> dist(fallback_lower_, fallback_upper_);
    return std::max(1, dist(rng));
}
