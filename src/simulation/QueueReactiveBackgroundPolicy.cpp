#include "simulation/QueueReactiveBackgroundPolicy.hpp"

#include "simulation/MultiAssetConfiguration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace dlob {
namespace {

constexpr std::array<const char*, background_hawkes_event_type_count>
event_names{{"limit_buy", "limit_sell", "market_buy", "market_sell",
             "cancel_bid", "cancel_ask"}};
constexpr std::array<const char*, background_hawkes_state_feature_count>
feature_names{{"log_spread_ratio", "log_bid_depth_ratio",
               "log_ask_depth_ratio", "queue_imbalance"}};

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream input(line);
    std::string field;
    while (std::getline(input, field, ',')) {
        if (!field.empty() && field.back() == '\r') field.pop_back();
        fields.push_back(std::move(field));
    }
    if (!line.empty() && line.back() == ',') fields.emplace_back();
    return fields;
}

std::size_t require_column(const std::vector<std::string>& header,
                           const char* name,
                           const std::filesystem::path& path) {
    const auto found = std::find(header.begin(), header.end(), name);
    if (found == header.end()) {
        throw std::runtime_error(path.string() + " is missing column " + name);
    }
    return static_cast<std::size_t>(std::distance(header.begin(), found));
}

double parse_double(const std::string& text, const char* label) {
    std::size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &used);
    } catch (...) {
        throw std::runtime_error(std::string("invalid ") + label + ": " + text);
    }
    if (used != text.size() || !std::isfinite(value)) {
        throw std::runtime_error(std::string("invalid ") + label + ": " + text);
    }
    return value;
}

long long parse_integer(const std::string& text, const char* label) {
    std::size_t used = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &used);
    } catch (...) {
        throw std::runtime_error(std::string("invalid ") + label + ": " + text);
    }
    if (used != text.size()) {
        throw std::runtime_error(std::string("invalid ") + label + ": " + text);
    }
    return value;
}

std::size_t named_index(const std::string& name,
                        const auto& names,
                        const char* label) {
    const auto found = std::find_if(
        names.begin(), names.end(),
        [&](const char* candidate) { return name == candidate; });
    if (found == names.end()) {
        throw std::runtime_error(std::string("unknown ") + label + ": " + name);
    }
    return static_cast<std::size_t>(std::distance(names.begin(), found));
}

std::filesystem::path resolve_relative(const std::filesystem::path& base,
                                       const std::string& value) {
    if (value.empty()) throw std::runtime_error("empty policy artifact path");
    const std::filesystem::path path(value);
    return path.is_absolute() ? path : (base / path).lexically_normal();
}

void validate_improvement_distribution(const std::filesystem::path& path,
                                       int tick_size) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "cannot open queue-policy improvement distribution: "
            + path.string());
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error(
            "empty queue-policy improvement distribution: " + path.string());
    }
    const auto header = split_csv(line);
    const std::size_t ticks_col = require_column(
        header, "improvement_ticks", path);
    const std::size_t units_col = require_column(
        header, "improvement_price_units", path);
    const std::size_t count_col = require_column(header, "count", path);
    const std::size_t required = 1U + std::max({ticks_col, units_col, count_col});
    bool observed = false;
    std::set<long long> distances;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split_csv(line);
        if (fields.size() < required) {
            throw std::runtime_error(
                "short row in queue-policy improvement distribution: "
                + path.string());
        }
        const double ticks = parse_double(fields[ticks_col], "improvement_ticks");
        const long long units = parse_integer(
            fields[units_col], "improvement_price_units");
        const long long count = parse_integer(fields[count_col], "improvement count");
        if (!(ticks > 0.0) || units <= 0 || count <= 0
            || units % tick_size != 0
            || std::abs(ticks - static_cast<double>(units) / tick_size)
                > 1.0e-12
            || !distances.insert(units).second) {
            throw std::runtime_error(
                "queue-policy improvement distribution is not an exact, "
                "positive tick-grid distribution: " + path.string());
        }
        observed = true;
    }
    if (!observed) {
        throw std::runtime_error(
            "queue-policy improvement distribution has no observations: "
            + path.string());
    }
}

BackgroundHawkesVector load_symbol_targets(const std::string& rates_file) {
    std::ifstream input(rates_file);
    if (!input) {
        throw std::runtime_error("cannot open symbol Hawkes rates: " + rates_file);
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("empty symbol Hawkes rates: " + rates_file);
    }
    const auto header = split_csv(line);
    const std::size_t event_col = require_column(header, "event_type", rates_file);
    const std::size_t target_col = require_column(
        header, "stationary_target_rate", rates_file);
    const std::size_t required = 1U + std::max(event_col, target_col);
    BackgroundHawkesVector targets{};
    std::array<bool, background_hawkes_event_type_count> seen{};
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split_csv(line);
        if (fields.size() < required) {
            throw std::runtime_error("short row in symbol Hawkes rates: " + rates_file);
        }
        const std::size_t event = named_index(fields[event_col], event_names, "event type");
        if (seen[event]) {
            throw std::runtime_error("duplicate event type in symbol Hawkes rates");
        }
        targets[event] = parse_double(fields[target_col], "stationary target rate");
        if (targets[event] < 0.0) {
            throw std::runtime_error("stationary target rates must be nonnegative");
        }
        seen[event] = true;
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        throw std::runtime_error("symbol Hawkes rates lack one or more event types");
    }
    return targets;
}

struct ClusterPolicy {
    int cluster_id = -1;
    double activity_scale = 1.0;
    double fast_beta = 0.0;
    double slow_beta = 0.0;
    double state_bound = 0.0;
    std::int64_t intraday_origin_ns = 0;
    std::int64_t intraday_bin_width_ns = 0;
    BackgroundHawkesMatrix fast_alpha{};
    BackgroundHawkesMatrix slow_alpha{};
    BackgroundStateCoefficientMatrix state_coefficients{};
    std::vector<BackgroundHawkesVector> intraday_factors;
};

double policy_spectral_radius(const ClusterPolicy& policy) {
    BackgroundHawkesMatrix integrated{};
    for (std::size_t row = 0; row < integrated.size(); ++row) {
        for (std::size_t column = 0; column < integrated[row].size(); ++column) {
            integrated[row][column] =
                policy.fast_alpha[row][column] / policy.fast_beta
                + policy.slow_alpha[row][column] / policy.slow_beta;
        }
    }
    BackgroundHawkesVector vector{};
    vector.fill(1.0);
    double estimate = 0.0;
    double previous = -1.0;
    for (int iteration = 0; iteration < 20'000; ++iteration) {
        BackgroundHawkesVector product{};
        for (std::size_t row = 0; row < integrated.size(); ++row) {
            product[row] = vector[row];
            for (std::size_t column = 0; column < integrated[row].size(); ++column) {
                product[row] += integrated[row][column] * vector[column];
            }
        }
        const double norm = *std::max_element(product.begin(), product.end());
        if (!(norm > 0.0) || !std::isfinite(norm)) {
            throw std::runtime_error("queue policy has invalid branching matrix");
        }
        for (double& value : product) value /= norm;
        vector = product;
        double numerator = 0.0;
        double denominator = 0.0;
        for (std::size_t row = 0; row < integrated.size(); ++row) {
            double shifted = vector[row];
            for (std::size_t column = 0; column < integrated[row].size(); ++column) {
                shifted += integrated[row][column] * vector[column];
            }
            numerator += vector[row] * shifted;
            denominator += vector[row] * vector[row];
        }
        estimate = numerator / denominator - 1.0;
        if (iteration > 64
            && std::abs(estimate - previous)
                <= 1.0e-14 * std::max(1.0, std::abs(estimate))) {
            break;
        }
        previous = estimate;
    }
    return std::max(0.0, estimate);
}

double maximum_integrated_row_sum(const ClusterPolicy& policy) {
    double maximum = 0.0;
    for (std::size_t row = 0; row < policy.fast_alpha.size(); ++row) {
        double sum = 0.0;
        for (std::size_t column = 0;
             column < policy.fast_alpha[row].size(); ++column) {
            sum += policy.fast_alpha[row][column] / policy.fast_beta
                + policy.slow_alpha[row][column] / policy.slow_beta;
        }
        maximum = std::max(maximum, sum);
    }
    return maximum;
}

double maximum_integrated_column_sum(const ClusterPolicy& policy) {
    double maximum = 0.0;
    for (std::size_t column = 0;
         column < policy.fast_alpha.size(); ++column) {
        double sum = 0.0;
        for (std::size_t row = 0;
             row < policy.fast_alpha.size(); ++row) {
            sum += policy.fast_alpha[row][column] / policy.fast_beta
                + policy.slow_alpha[row][column] / policy.slow_beta;
        }
        maximum = std::max(maximum, sum);
    }
    return maximum;
}

bool has_state_response(const ClusterPolicy& policy) {
    for (const auto& row : policy.state_coefficients) {
        for (const double value : row) {
            if (value != 0.0) return true;
        }
    }
    return false;
}

ClusterPolicy load_cluster_policy(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open queue policy: " + path.string());
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty queue policy");
    const auto header = split_csv(line);
    const std::size_t kind_col = require_column(header, "kind", path);
    const std::size_t target_col = require_column(header, "target", path);
    const std::size_t source_col = require_column(header, "source", path);
    const std::size_t bin_col = require_column(header, "bin", path);
    const std::size_t value_col = require_column(header, "value", path);
    const std::size_t required = 1U + std::max({
        kind_col, target_col, source_col, bin_col, value_col});

    ClusterPolicy policy;
    std::map<std::string, double> metadata;
    std::map<std::string, std::string> text_metadata;
    std::array<std::array<bool, background_hawkes_event_type_count>,
               background_hawkes_event_type_count> fast_seen{};
    auto slow_seen = fast_seen;
    std::array<std::array<bool, background_hawkes_state_feature_count>,
               background_hawkes_event_type_count> state_seen{};
    std::map<int, BackgroundHawkesVector> intraday;
    std::map<int, std::array<bool, background_hawkes_event_type_count>>
        intraday_seen;

    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        const auto fields = split_csv(line);
        if (fields.size() < required) {
            throw std::runtime_error("short queue-policy row in " + path.string());
        }
        const std::string& kind = fields[kind_col];
        if (kind == "meta") {
            const std::string& key = fields[target_col];
            if (key == "matrix_orientation" || key == "stationary_target_scope") {
                if (!text_metadata.emplace(key, fields[value_col]).second) {
                    throw std::runtime_error("duplicate queue-policy text metadata key");
                }
                continue;
            }
            const double value = parse_double(
                fields[value_col], "queue-policy metadata value");
            if (!metadata.emplace(key, value).second) {
                throw std::runtime_error("duplicate queue-policy numeric metadata key");
            }
            continue;
        }
        const double value = parse_double(fields[value_col], "queue-policy value");
        if (kind == "fast_alpha" || kind == "slow_alpha") {
            const std::size_t target = named_index(
                fields[target_col], event_names, "target event");
            const std::size_t source = named_index(
                fields[source_col], event_names, "source event");
            auto& matrix = kind == "fast_alpha"
                ? policy.fast_alpha : policy.slow_alpha;
            auto& seen = kind == "fast_alpha" ? fast_seen : slow_seen;
            if (seen[target][source] || value < 0.0) {
                throw std::runtime_error("duplicate or negative Hawkes matrix entry");
            }
            matrix[target][source] = value;
            seen[target][source] = true;
        } else if (kind == "state_coefficient") {
            const std::size_t target = named_index(
                fields[target_col], event_names, "target event");
            const std::size_t feature = named_index(
                fields[source_col], feature_names, "state feature");
            if (state_seen[target][feature]) {
                throw std::runtime_error("duplicate state coefficient");
            }
            policy.state_coefficients[target][feature] = value;
            state_seen[target][feature] = true;
        } else if (kind == "intraday_factor") {
            const std::size_t target = named_index(
                fields[target_col], event_names, "target event");
            const long long parsed_bin = parse_integer(fields[bin_col], "intraday bin");
            if (parsed_bin < 0 || parsed_bin > 1'000'000 || value < 0.0) {
                throw std::runtime_error("invalid intraday factor row");
            }
            const int bin = static_cast<int>(parsed_bin);
            if (intraday_seen[bin][target]) {
                throw std::runtime_error("duplicate intraday factor");
            }
            intraday[bin][target] = value;
            intraday_seen[bin][target] = true;
        } else if (kind == "diagnostic_cluster_target") {
            // Written for audit only. Runtime targets remain symbol-specific.
            (void)named_index(fields[target_col], event_names, "target event");
        } else {
            throw std::runtime_error("unknown queue-policy row kind: " + kind);
        }
    }

    const auto meta = [&](const char* name) {
        const auto found = metadata.find(name);
        if (found == metadata.end()) {
            throw std::runtime_error(std::string("queue policy lacks meta ") + name);
        }
        return found->second;
    };
    const double schema = meta("schema_version");
    if (schema != 1.0) throw std::runtime_error("unsupported queue-policy schema");
    const auto orientation = text_metadata.find("matrix_orientation");
    const auto target_scope = text_metadata.find("stationary_target_scope");
    if (orientation == text_metadata.end()
        || orientation->second != "response_rows_trigger_columns"
        || target_scope == text_metadata.end()
        || target_scope->second != "descriptive_cluster_member_mean") {
        throw std::runtime_error("queue policy has incompatible matrix or target semantics");
    }
    policy.activity_scale = meta("activity_scale");
    const double cluster_value = meta("cluster_id");
    if (cluster_value < 0.0 || cluster_value > std::numeric_limits<int>::max()
        || cluster_value != std::floor(cluster_value)) {
        throw std::runtime_error("queue policy has invalid cluster_id metadata");
    }
    policy.cluster_id = static_cast<int>(cluster_value);
    policy.fast_beta = metadata.contains("fast_beta")
        ? meta("fast_beta") : meta("fast_beta_per_second");
    policy.slow_beta = metadata.contains("slow_beta")
        ? meta("slow_beta") : meta("slow_beta_per_second");
    policy.state_bound = meta("state_log_multiplier_bound");
    policy.intraday_origin_ns = static_cast<std::int64_t>(
        std::llround(meta("intraday_origin_ns")));
    policy.intraday_bin_width_ns = static_cast<std::int64_t>(
        std::llround(meta("intraday_bin_width_ns")));
    if (!(policy.activity_scale > 0.0) || !(policy.fast_beta > 0.0)
        || !(policy.slow_beta > 0.0) || !(policy.state_bound > 0.0)
        || policy.intraday_origin_ns < 0 || policy.intraday_bin_width_ns <= 0) {
        throw std::runtime_error("invalid queue-policy metadata");
    }
    const double declared_radius = meta("spectral_radius");
    if (declared_radius < 0.0 || declared_radius >= 0.75) {
        throw std::runtime_error(
            "queue-policy declared spectral radius must be below 0.75");
    }
    for (std::size_t row = 0; row < fast_seen.size(); ++row) {
        if (std::find(fast_seen[row].begin(), fast_seen[row].end(), false)
                != fast_seen[row].end()
            || std::find(slow_seen[row].begin(), slow_seen[row].end(), false)
                != slow_seen[row].end()
            || std::find(state_seen[row].begin(), state_seen[row].end(), false)
                != state_seen[row].end()) {
            throw std::runtime_error("queue policy has an incomplete matrix");
        }
    }
    if (intraday.empty() || intraday.begin()->first != 0
        || intraday.rbegin()->first + 1 != static_cast<int>(intraday.size())) {
        throw std::runtime_error("queue policy intraday bins must be contiguous from zero");
    }
    for (const auto& [bin, values] : intraday) {
        (void)values;
        if (std::find(intraday_seen[bin].begin(), intraday_seen[bin].end(), false)
            != intraday_seen[bin].end()) {
            throw std::runtime_error("queue policy has an incomplete intraday bin");
        }
        policy.intraday_factors.push_back(intraday.at(bin));
    }
    const double actual_radius = policy_spectral_radius(policy);
    // The exact Perron root of a defective sparse matrix can converge only
    // algebraically under power iteration.  A strict maximum-row-sum bound is
    // an exact dependency-free stability certificate; the declared-radius
    // comparison remains a provenance check with a documented numerical
    // tolerance rather than the primary safety gate.
    const double certified_row_bound = maximum_integrated_row_sum(policy);
    const double certified_column_bound =
        maximum_integrated_column_sum(policy);
    const double radius_tolerance = 1.0e-4
        * std::max({1.0, std::abs(actual_radius), std::abs(declared_radius)});
    if (certified_row_bound >= 0.75
        || (has_state_response(policy) && certified_column_bound >= 0.75)
        || std::abs(actual_radius - declared_radius) > radius_tolerance) {
        throw std::runtime_error(
            "queue policy branching matrix disagrees with declared spectral radius");
    }
    return policy;
}

struct MappingRow {
    int cluster_id = -1;
    std::filesystem::path policy_file;
    std::filesystem::path buy_improvement_file;
    std::filesystem::path sell_improvement_file;
};

} // namespace

QueueReactiveBackgroundBundle load_queue_reactive_background_bundle(
    const std::filesystem::path& mapping_csv,
    const std::vector<MultiAssetBookConfig>& assets,
    std::uint64_t simulation_seed,
    int tick_size) {
    if (assets.empty() || tick_size <= 0) {
        throw std::invalid_argument("queue-reactive bundle requires assets and tick size");
    }
    std::ifstream input(mapping_csv);
    if (!input) {
        throw std::runtime_error("cannot open queue-policy mapping: "
                                 + mapping_csv.string());
    }
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty policy mapping");
    const auto header = split_csv(line);
    const std::size_t symbol_col = require_column(header, "symbol", mapping_csv);
    const std::size_t cluster_col = require_column(header, "cluster_id", mapping_csv);
    const std::size_t policy_col = require_column(header, "policy_file", mapping_csv);
    const std::size_t buy_col = require_column(
        header, "limit_buy_improvement_file", mapping_csv);
    const std::size_t sell_col = require_column(
        header, "limit_sell_improvement_file", mapping_csv);
    const std::size_t required = 1U + std::max({
        symbol_col, cluster_col, policy_col, buy_col, sell_col});
    std::unordered_map<std::string, MappingRow> rows;
    const std::filesystem::path base = std::filesystem::absolute(mapping_csv)
        .parent_path();
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split_csv(line);
        if (fields.size() < required || fields[symbol_col].empty()) {
            throw std::runtime_error("invalid queue-policy mapping row");
        }
        const long long cluster = parse_integer(fields[cluster_col], "cluster_id");
        if (cluster < 0 || cluster > std::numeric_limits<int>::max()) {
            throw std::runtime_error("invalid queue-policy cluster id");
        }
        MappingRow row;
        row.cluster_id = static_cast<int>(cluster);
        row.policy_file = resolve_relative(base, fields[policy_col]);
        row.buy_improvement_file = resolve_relative(base, fields[buy_col]);
        row.sell_improvement_file = resolve_relative(base, fields[sell_col]);
        if (!rows.emplace(fields[symbol_col], std::move(row)).second) {
            throw std::runtime_error("duplicate symbol in queue-policy mapping");
        }
    }
    if (rows.size() != assets.size()) {
        throw std::runtime_error(
            "queue-policy mapping must contain exactly one row per asset");
    }

    std::set<std::filesystem::path> validated_improvement_files;
    for (const auto& [symbol, row] : rows) {
        (void)symbol;
        for (const std::filesystem::path& path : {
                 row.buy_improvement_file, row.sell_improvement_file}) {
            if (validated_improvement_files.insert(path).second) {
                validate_improvement_distribution(path, tick_size);
            }
        }
    }

    std::map<std::filesystem::path, ClusterPolicy> cache;
    QueueReactiveBackgroundBundle bundle;
    bundle.configs.reserve(assets.size());
    bundle.cluster_ids.reserve(assets.size());
    for (std::size_t index = 0; index < assets.size(); ++index) {
        const MultiAssetBookConfig& asset = assets[index];
        const auto row_it = rows.find(asset.symbol);
        if (row_it == rows.end()) {
            throw std::runtime_error(
                "queue-policy mapping is missing symbol " + asset.symbol);
        }
        const MappingRow& mapping = row_it->second;
        const auto [policy_it, inserted] = cache.try_emplace(mapping.policy_file);
        if (inserted) policy_it->second = load_cluster_policy(mapping.policy_file);
        const ClusterPolicy& policy = policy_it->second;
        if (policy.cluster_id != mapping.cluster_id) {
            throw std::runtime_error(
                "queue-policy mapping cluster disagrees with policy metadata");
        }

        BackgroundHawkesConfig config = make_multi_asset_background_config(
            asset, static_cast<BookId>(index), simulation_seed, tick_size);
        config.activity_scale = policy.activity_scale;
        config.beta = policy.fast_beta;
        config.slow_beta = policy.slow_beta;
        config.alpha = policy.fast_alpha;
        config.slow_alpha = policy.slow_alpha;
        config.state_log_multiplier_coefficients = policy.state_coefficients;
        config.state_reference_bid_depth = asset.target_mean_bid_depth;
        config.state_reference_ask_depth = asset.target_mean_ask_depth;
        config.state_log_multiplier_bound = policy.state_bound;
        config.intraday_origin_ns = policy.intraday_origin_ns;
        config.intraday_bin_width_ns = policy.intraday_bin_width_ns;
        config.intraday_factors = policy.intraday_factors;
        config.limit_buy_improvement_file =
            mapping.buy_improvement_file.string();
        config.limit_sell_improvement_file =
            mapping.sell_improvement_file.string();
        // The fitted conditional-multinomial policy redistributes event-type
        // hazard while preserving its total.  It does not model cancellation
        // mark size.  Retain the separately identified, bounded depth response
        // for cancellation quantities so the reduced book has a restoring
        // depletion mechanism rather than accumulating unmatched limit marks.
        config.cancellation_quantity_depth_scaling = true;
        config.stationary_target_rates = load_symbol_targets(
            asset.hawkes_rates_file);

        for (std::size_t target = 0; target < config.mu.size(); ++target) {
            double endogenous = 0.0;
            for (std::size_t source = 0; source < config.mu.size(); ++source) {
                endogenous += (
                    config.alpha[target][source] / config.beta
                    + config.slow_alpha[target][source] / config.slow_beta)
                    * config.stationary_target_rates[source];
            }
            const double immigration =
                (config.stationary_target_rates[target] - endogenous)
                / config.activity_scale;
            const double tolerance = 1.0e-12 * std::max(
                1.0, config.stationary_target_rates[target]);
            if (immigration < -tolerance || !std::isfinite(immigration)) {
                throw std::runtime_error(
                    "queue policy implies negative immigration for "
                    + asset.symbol + " event " + event_names[target]);
            }
            config.mu[target] = std::max(0.0, immigration);
        }
        config.validate_stationary_target = true;
        // Construction performs the complete stability/stationarity audit.
        const BackgroundHawkesStream audit(config);
        (void)audit;
        bundle.configs.push_back(std::move(config));
        bundle.cluster_ids.push_back(mapping.cluster_id);
    }
    return bundle;
}

} // namespace dlob
