#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct AgentInput {
    std::uint32_t kind;
    std::uint32_t book;
    std::int32_t threshold_ticks;
    std::int32_t quantity;
    std::int32_t bias_ticks;
    std::int32_t max_inventory;
    std::int32_t inventory;
    std::uint32_t padding;
};

struct MarketSnapshot {
    std::int32_t best_bid_ticks;
    std::int32_t best_ask_ticks;
    std::int32_t fundamental_ticks;
    std::int32_t previous_mid_ticks;
};

struct OrderIntent {
    std::int32_t bid_price_ticks;
    std::int32_t ask_price_ticks;
    std::int32_t buy_quantity;
    std::int32_t sell_quantity;
};

static_assert(sizeof(AgentInput) == 32);
static_assert(sizeof(MarketSnapshot) == 16);
static_assert(sizeof(OrderIntent) == 16);

constexpr const char* metal_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct AgentInput {
    uint kind;
    uint book;
    int threshold_ticks;
    int quantity;
    int bias_ticks;
    int max_inventory;
    int inventory;
    uint padding;
};

struct MarketSnapshot {
    int best_bid_ticks;
    int best_ask_ticks;
    int fundamental_ticks;
    int previous_mid_ticks;
};

struct OrderIntent {
    int bid_price_ticks;
    int ask_price_ticks;
    int buy_quantity;
    int sell_quantity;
};

kernel void agent_decisions(
    device const AgentInput* agents [[buffer(0)]],
    device const MarketSnapshot* markets [[buffer(1)]],
    device OrderIntent* output [[buffer(2)]],
    constant uint& agent_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= agent_count) return;

    const AgentInput agent = agents[gid];
    const MarketSnapshot market = markets[agent.book];
    const int mid = (market.best_bid_ticks + market.best_ask_ticks) / 2;
    OrderIntent intent = {0, 0, 0, 0};

    if (agent.kind == 0u) {
        // Passive market maker: quote both sides with an inventory skew.
        if (abs(agent.inventory) < agent.max_inventory) {
            const int skew = agent.inventory > 0 ? -1 : (agent.inventory < 0 ? 1 : 0);
            intent.bid_price_ticks = market.best_bid_ticks + skew;
            intent.ask_price_ticks = market.best_ask_ticks + skew;
            intent.buy_quantity = agent.quantity;
            intent.sell_quantity = agent.quantity;
        }
    } else if (agent.kind == 1u) {
        // Momentum trader: cross the spread in the direction of the last move.
        const int move = mid - market.previous_mid_ticks;
        if (abs(move) >= agent.threshold_ticks) {
            if (move > 0) {
                intent.ask_price_ticks = market.best_ask_ticks;
                intent.buy_quantity = agent.quantity;
            } else {
                intent.bid_price_ticks = market.best_bid_ticks;
                intent.sell_quantity = agent.quantity;
            }
        }
    } else if (agent.kind == 2u) {
        // Informed/value trader: compare a private value with the current mid.
        const int signal = market.fundamental_ticks + agent.bias_ticks - mid;
        if (abs(signal) >= agent.threshold_ticks) {
            if (signal > 0) {
                intent.ask_price_ticks = market.best_ask_ticks;
                intent.buy_quantity = agent.quantity;
            } else {
                intent.bid_price_ticks = market.best_bid_ticks;
                intent.sell_quantity = agent.quantity;
            }
        }
    } else {
        // Institutional child order: the sign of bias_ticks fixes its side.
        if (agent.bias_ticks >= 0) {
            intent.ask_price_ticks = market.best_ask_ticks;
            intent.buy_quantity = agent.quantity;
        } else {
            intent.bid_price_ticks = market.best_bid_ticks;
            intent.sell_quantity = agent.quantity;
        }
    }
    output[gid] = intent;
}
)METAL";

[[nodiscard]] OrderIntent decide_cpu(const AgentInput& agent,
                                     const MarketSnapshot& market) {
    const std::int32_t mid = (market.best_bid_ticks + market.best_ask_ticks) / 2;
    OrderIntent intent{};
    if (agent.kind == 0U) {
        if (std::abs(agent.inventory) < agent.max_inventory) {
            const std::int32_t skew = agent.inventory > 0 ? -1 : (agent.inventory < 0 ? 1 : 0);
            intent.bid_price_ticks = market.best_bid_ticks + skew;
            intent.ask_price_ticks = market.best_ask_ticks + skew;
            intent.buy_quantity = agent.quantity;
            intent.sell_quantity = agent.quantity;
        }
    } else if (agent.kind == 1U) {
        const std::int32_t move = mid - market.previous_mid_ticks;
        if (std::abs(move) >= agent.threshold_ticks) {
            if (move > 0) {
                intent.ask_price_ticks = market.best_ask_ticks;
                intent.buy_quantity = agent.quantity;
            } else {
                intent.bid_price_ticks = market.best_bid_ticks;
                intent.sell_quantity = agent.quantity;
            }
        }
    } else if (agent.kind == 2U) {
        const std::int32_t signal = market.fundamental_ticks + agent.bias_ticks - mid;
        if (std::abs(signal) >= agent.threshold_ticks) {
            if (signal > 0) {
                intent.ask_price_ticks = market.best_ask_ticks;
                intent.buy_quantity = agent.quantity;
            } else {
                intent.bid_price_ticks = market.best_bid_ticks;
                intent.sell_quantity = agent.quantity;
            }
        }
    } else if (agent.bias_ticks >= 0) {
        intent.ask_price_ticks = market.best_ask_ticks;
        intent.buy_quantity = agent.quantity;
    } else {
        intent.bid_price_ticks = market.best_bid_ticks;
        intent.sell_quantity = agent.quantity;
    }
    return intent;
}

[[nodiscard]] std::size_t parse_size(const char* text, const char* label) {
    const std::string value(text);
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || parsed == 0ULL
        || parsed > static_cast<unsigned long long>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + value);
    }
    return static_cast<std::size_t>(parsed);
}

[[nodiscard]] bool equal_intent(const OrderIntent& lhs, const OrderIntent& rhs) {
    return lhs.bid_price_ticks == rhs.bid_price_ticks
        && lhs.ask_price_ticks == rhs.ask_price_ticks
        && lhs.buy_quantity == rhs.buy_quantity
        && lhs.sell_quantity == rhs.sell_quantity;
}

[[nodiscard]] std::uint64_t hash_outputs(const std::vector<OrderIntent>& outputs) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const OrderIntent& output : outputs) {
        const std::int32_t values[] = {output.bid_price_ticks, output.ask_price_ticks,
                                       output.buy_quantity, output.sell_quantity};
        for (const std::int32_t value : values) {
            const auto bits = static_cast<std::uint32_t>(value);
            for (int byte = 0; byte < 4; ++byte) {
                hash ^= (bits >> static_cast<unsigned>(byte * 8)) & 0xffU;
                hash *= 1099511628211ULL;
            }
        }
    }
    return hash;
}

[[nodiscard]] std::vector<AgentInput> make_agents(std::size_t count,
                                                   std::size_t book_count) {
    std::vector<AgentInput> agents(count);
    for (std::size_t index = 0; index < count; ++index) {
        AgentInput& agent = agents[index];
        agent.kind = static_cast<std::uint32_t>(index % 4U);
        agent.book = static_cast<std::uint32_t>((index / 4U) % book_count);
        agent.threshold_ticks = 1 + static_cast<std::int32_t>(index % 5U);
        agent.quantity = 10 + static_cast<std::int32_t>(index % 100U);
        agent.bias_ticks = static_cast<std::int32_t>(index % 17U) - 8;
        agent.max_inventory = 1'000;
        agent.inventory = static_cast<std::int32_t>(index % 401U) - 200;
        agent.padding = 0U;
    }
    return agents;
}

[[nodiscard]] std::vector<MarketSnapshot> make_markets(std::size_t steps,
                                                        std::size_t book_count) {
    std::vector<MarketSnapshot> markets(steps * book_count);
    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t book = 0; book < book_count; ++book) {
            const std::int32_t base = 10'000 + static_cast<std::int32_t>(book * 1'000U);
            const std::int32_t drift = static_cast<std::int32_t>((step * 7U + book * 3U) % 23U) - 11;
            const std::int32_t previous_drift =
                static_cast<std::int32_t>(((step + 22U) * 7U + book * 3U) % 23U) - 11;
            MarketSnapshot& market = markets[step * book_count + book];
            market.best_bid_ticks = base + drift;
            market.best_ask_ticks = market.best_bid_ticks
                + 2 + static_cast<std::int32_t>((step + book) % 3U);
            market.fundamental_ticks = base
                + static_cast<std::int32_t>((step * 3U + book * 7U) % 31U) - 15;
            market.previous_mid_ticks = base + previous_drift + 1;
        }
    }
    return markets;
}

template <typename Clock = std::chrono::steady_clock>
[[nodiscard]] double elapsed_seconds(typename Clock::time_point start,
                                     typename Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            std::size_t agent_count = 262'144;
            std::size_t step_count = 256;
            std::size_t book_count = 4;
            bool cpu_only = false;
            for (int index = 1; index < argc; ++index) {
                const std::string option(argv[index]);
                if (option == "--cpu-only") {
                    cpu_only = true;
                    continue;
                }
                if (index + 1 >= argc) {
                    throw std::invalid_argument("missing value for " + option);
                }
                if (option == "--agents") {
                    agent_count = parse_size(argv[++index], "agent count");
                } else if (option == "--steps") {
                    step_count = parse_size(argv[++index], "step count");
                } else if (option == "--books") {
                    book_count = parse_size(argv[++index], "book count");
                } else {
                    throw std::invalid_argument("unknown option: " + option);
                }
            }
            if (book_count > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw std::invalid_argument("book count exceeds GPU index range");
            }
            if (step_count > std::numeric_limits<std::size_t>::max() / book_count) {
                throw std::invalid_argument("market history size overflow");
            }

            const std::vector<AgentInput> agents = make_agents(agent_count, book_count);
            const std::vector<MarketSnapshot> markets = make_markets(step_count, book_count);
            std::vector<OrderIntent> cpu_output(agent_count);

            const auto cpu_start = std::chrono::steady_clock::now();
            for (std::size_t step = 0; step < step_count; ++step) {
                const MarketSnapshot* step_markets = markets.data() + step * book_count;
                for (std::size_t agent = 0; agent < agent_count; ++agent) {
                    cpu_output[agent] = decide_cpu(agents[agent], step_markets[agents[agent].book]);
                }
            }
            const auto cpu_end = std::chrono::steady_clock::now();
            const double cpu_seconds = elapsed_seconds(cpu_start, cpu_end);
            const long double decisions = static_cast<long double>(agent_count)
                * static_cast<long double>(step_count);
            const double cpu_rate = static_cast<double>(decisions / cpu_seconds);

            if (cpu_only) {
                std::cout << std::setprecision(9)
                          << "device=cpu-only\n"
                          << "agents=" << agent_count << '\n'
                          << "books=" << book_count << '\n'
                          << "steps=" << step_count << '\n'
                          << "decisions=" << static_cast<unsigned long long>(decisions) << '\n'
                          << "cpu_seconds=" << cpu_seconds << '\n'
                          << "cpu_decisions_per_second=" << cpu_rate << '\n'
                          << "cpu_output_hash=" << hash_outputs(cpu_output) << '\n';
                return EXIT_SUCCESS;
            }

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) throw std::runtime_error("Metal device is unavailable");

            NSError* compile_error = nil;
            const auto compile_start = std::chrono::steady_clock::now();
            NSString* source = [NSString stringWithUTF8String:metal_source];
            id<MTLLibrary> library = [device newLibraryWithSource:source
                                                          options:nil
                                                            error:&compile_error];
            if (library == nil) {
                const char* message = compile_error == nil
                    ? "unknown Metal compilation error"
                    : compile_error.localizedDescription.UTF8String;
                throw std::runtime_error(std::string("Metal compilation failed: ") + message);
            }
            id<MTLFunction> function = [library newFunctionWithName:@"agent_decisions"];
            if (function == nil) throw std::runtime_error("Metal kernel function is unavailable");
            NSError* pipeline_error = nil;
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function error:&pipeline_error];
            if (pipeline == nil) {
                const char* message = pipeline_error == nil
                    ? "unknown pipeline error"
                    : pipeline_error.localizedDescription.UTF8String;
                throw std::runtime_error(std::string("Metal pipeline creation failed: ") + message);
            }
            const auto compile_end = std::chrono::steady_clock::now();
            const double compile_seconds = elapsed_seconds(compile_start, compile_end);

            const NSUInteger agent_bytes = static_cast<NSUInteger>(agents.size() * sizeof(AgentInput));
            const NSUInteger market_bytes = static_cast<NSUInteger>(markets.size() * sizeof(MarketSnapshot));
            const NSUInteger output_bytes = static_cast<NSUInteger>(agent_count * sizeof(OrderIntent));
            id<MTLBuffer> agent_buffer = [device newBufferWithBytes:agents.data()
                                                            length:agent_bytes
                                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> market_buffer = [device newBufferWithBytes:markets.data()
                                                             length:market_bytes
                                                            options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buffer = [device newBufferWithLength:output_bytes
                                                               options:MTLResourceStorageModeShared];
            if (agent_buffer == nil || market_buffer == nil || output_buffer == nil) {
                throw std::runtime_error("Metal buffer allocation failed");
            }
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (queue == nil) throw std::runtime_error("Metal command queue creation failed");

            const NSUInteger width = pipeline.threadExecutionWidth;
            const MTLSize threads_per_group = MTLSizeMake(width, 1, 1);
            const MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(agent_count), 1, 1);
            const std::uint32_t gpu_agent_count = static_cast<std::uint32_t>(agent_count);

            // Warm-up excludes one-time driver setup from the measured kernel loop.
            @autoreleasepool {
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:agent_buffer offset:0 atIndex:0];
                [encoder setBuffer:market_buffer offset:0 atIndex:1];
                [encoder setBuffer:output_buffer offset:0 atIndex:2];
                [encoder setBytes:&gpu_agent_count length:sizeof(gpu_agent_count) atIndex:3];
                [encoder dispatchThreads:grid threadsPerThreadgroup:threads_per_group];
                [encoder endEncoding];
                [command commit];
                [command waitUntilCompleted];
                if (command.status == MTLCommandBufferStatusError) {
                    throw std::runtime_error("Metal warm-up dispatch failed");
                }
            }

            const auto gpu_start = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:agent_buffer offset:0 atIndex:0];
            [encoder setBuffer:output_buffer offset:0 atIndex:2];
            [encoder setBytes:&gpu_agent_count length:sizeof(gpu_agent_count) atIndex:3];
            for (std::size_t step = 0; step < step_count; ++step) {
                const NSUInteger offset = static_cast<NSUInteger>(step * book_count
                                                                   * sizeof(MarketSnapshot));
                [encoder setBuffer:market_buffer offset:offset atIndex:1];
                [encoder dispatchThreads:grid threadsPerThreadgroup:threads_per_group];
            }
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            const auto gpu_end = std::chrono::steady_clock::now();
            if (command.status == MTLCommandBufferStatusError) {
                const char* message = command.error == nil
                    ? "unknown command-buffer error"
                    : command.error.localizedDescription.UTF8String;
                throw std::runtime_error(std::string("Metal benchmark failed: ") + message);
            }
            const double gpu_seconds = elapsed_seconds(gpu_start, gpu_end);

            std::vector<OrderIntent> gpu_output(agent_count);
            std::copy_n(static_cast<const OrderIntent*>(output_buffer.contents), agent_count,
                        gpu_output.begin());
            std::size_t mismatch_count = 0;
            std::size_t first_mismatch = agent_count;
            for (std::size_t index = 0; index < agent_count; ++index) {
                if (!equal_intent(cpu_output[index], gpu_output[index])) {
                    if (first_mismatch == agent_count) first_mismatch = index;
                    ++mismatch_count;
                }
            }

            const double gpu_rate = static_cast<double>(decisions / gpu_seconds);
            std::cout << std::setprecision(9)
                      << "device=" << device.name.UTF8String << '\n'
                      << "agents=" << agent_count << '\n'
                      << "books=" << book_count << '\n'
                      << "steps=" << step_count << '\n'
                      << "decisions=" << static_cast<unsigned long long>(decisions) << '\n'
                      << "metal_compile_seconds=" << compile_seconds << '\n'
                      << "cpu_seconds=" << cpu_seconds << '\n'
                      << "gpu_seconds=" << gpu_seconds << '\n'
                      << "cpu_decisions_per_second=" << cpu_rate << '\n'
                      << "gpu_decisions_per_second=" << gpu_rate << '\n'
                      << "gpu_speedup=" << (cpu_seconds / gpu_seconds) << '\n'
                      << "cpu_output_hash=" << hash_outputs(cpu_output) << '\n'
                      << "gpu_output_hash=" << hash_outputs(gpu_output) << '\n'
                      << "mismatches=" << mismatch_count << '\n';
            if (first_mismatch != agent_count) {
                std::cerr << "first mismatch at agent " << first_mismatch << '\n';
            }
            return mismatch_count == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
        } catch (const std::exception& error) {
            std::cerr << "gpu_agent_benchmark: " << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
}
