#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>
#include <cstdint>

namespace shallenge {

// Shared state across all GPUs
struct SharedState {
    std::mutex best_hash_mutex;
    std::vector<uint32_t> best_hash;
    std::string best_nonce;

    std::atomic<bool> running{true};

    std::string username;
    std::chrono::steady_clock::time_point start_time;
};

} // namespace shallenge
