// Project code developed for Peter Zhang's thesis with OpenAI assistance; see PROVENANCE.md.
#include "calibration/CalibrationParameters.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace dlob::calibration {
namespace {

const std::array<const char*, parameter_count> names{
    "market_maker_interval_ms",
    "market_maker_min_spread_ticks",
    "momentum_rate_per_second",
    "momentum_threshold_ticks",
    "informed_rate_per_second",
    "informed_signal_precision",
    "institutional_rate_per_second",
    "institutional_participation_cap"
};

std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    std::size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) ++first;
    value.erase(0, first);
    return value;
}

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream stream(line);
    std::string cell;
    while (std::getline(stream, cell, ',')) out.push_back(trim(cell));
    return out;
}

std::array<ParameterSpec, parameter_count> defaults() {
    return {{
        {names[0], 5.0, 250.0, ParameterKind::Continuous},
        {names[1], 1.0, 10.0, ParameterKind::Integer},
        {names[2], 0.01, 0.50, ParameterKind::Continuous},
        {names[3], 0.05, 2.00, ParameterKind::Continuous},
        {names[4], 0.002, 0.20, ParameterKind::Continuous},
        {names[5], 0.20, 5.00, ParameterKind::Continuous},
        {names[6], 0.0005, 0.05, ParameterKind::Continuous},
        {names[7], 0.01, 0.30, ParameterKind::Continuous}
    }};
}

double map_value(const ParameterSpec& spec, double unit) {
    const double x = std::clamp(unit, 0.0, 1.0);
    const double value = spec.lower + x * (spec.upper - spec.lower);
    return spec.kind == ParameterKind::Integer ? std::round(value) : value;
}

} // namespace

const std::array<const char*, parameter_count>& parameter_names() { return names; }

ParameterSpace::ParameterSpace() : specs_(defaults()) {}

ParameterSpace::ParameterSpace(std::vector<ParameterSpec> specs) {
    if (specs.size() != parameter_count) {
        throw std::invalid_argument("ParameterSpace requires exactly 8 specifications");
    }
    for (std::size_t i = 0; i < parameter_count; ++i) {
        if (specs[i].name != names[i]) {
            throw std::invalid_argument(
                "Parameter specification row " + std::to_string(i)
                + " must be named '" + names[i] + "'");
        }
        if (!(specs[i].lower < specs[i].upper)) {
            throw std::invalid_argument("Parameter bounds must satisfy lower < upper for " + specs[i].name);
        }
        specs_[i] = std::move(specs[i]);
    }
}

ParameterSpace ParameterSpace::load_csv(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open parameter-space CSV: " + path.string());
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Empty parameter-space CSV: " + path.string());
    std::vector<ParameterSpec> specs;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        const auto fields = split(line);
        if (fields.size() < 4) throw std::runtime_error("Malformed parameter-space row: " + line);
        ParameterSpec spec;
        spec.name = fields[0];
        spec.lower = std::stod(fields[1]);
        spec.upper = std::stod(fields[2]);
        if (fields[3] == "integer" || fields[3] == "int") spec.kind = ParameterKind::Integer;
        else if (fields[3] == "continuous" || fields[3] == "double") spec.kind = ParameterKind::Continuous;
        else throw std::runtime_error("Unknown parameter kind in row: " + line);
        specs.push_back(std::move(spec));
    }
    return ParameterSpace(std::move(specs));
}

void ParameterSpace::write_default_csv(const std::filesystem::path& path) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot write parameter-space CSV: " + path.string());
    output << "name,lower,upper,type\n";
    for (const auto& spec : defaults()) {
        output << spec.name << ',' << spec.lower << ',' << spec.upper << ','
               << (spec.kind == ParameterKind::Integer ? "integer" : "continuous") << '\n';
    }
}

UnitParameters ParameterSpace::clamp(const UnitParameters& unit) const {
    UnitParameters out{};
    for (std::size_t i = 0; i < parameter_count; ++i) out[i] = std::clamp(unit[i], 0.0, 1.0);
    return out;
}

PhysicalParameters ParameterSpace::decode(const UnitParameters& unit) const {
    std::array<double, parameter_count> value{};
    for (std::size_t i = 0; i < parameter_count; ++i) value[i] = map_value(specs_[i], unit[i]);
    PhysicalParameters p;
    p.market_maker_interval_ms = value[0];
    p.market_maker_min_spread_ticks = static_cast<int>(value[1]);
    p.momentum_rate_per_second = value[2];
    p.momentum_threshold_ticks = value[3];
    p.informed_rate_per_second = value[4];
    p.informed_signal_precision = value[5];
    p.informed_signal_noise_ticks = 1.0 / std::max(1e-6, p.informed_signal_precision);
    p.institutional_rate_per_second = value[6];
    p.institutional_participation_cap = value[7];
    return p;
}

} // namespace dlob::calibration
