#pragma once

#include "common/DistributedTypes.hpp"
#include "common/PerformanceMetrics.hpp"

#include <vector>

namespace dlob {

std::vector<OrderMessage> gather_orders(const std::vector<OrderMessage>& local_orders,
                                        int rank,
                                        int world_size,
                                        PerformanceMetrics& metrics);

std::vector<AgentReport> scatter_reports(const std::vector<AgentReport>& root_reports,
                                         int rank,
                                         int world_size,
                                         PerformanceMetrics& metrics);

} // namespace dlob
