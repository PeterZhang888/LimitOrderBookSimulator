#pragma once

#include "common/AgentUtilities.hpp"

#include <string>
#include <vector>

namespace dlob {

class EmpiricalDistribution {
public:
    void set_fallback(int lower, int upper);
    bool load_from_csv(const std::string& filename, const std::string& column_name);
    int sample(FastRng& rng) const;

private:
    std::vector<int> values_;
    int fallback_lower_ = 1;
    int fallback_upper_ = 1;
};

} // namespace dlob
