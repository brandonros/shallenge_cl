#include "config.hpp"
#include "core/hash_utils.hpp"
#include "core/types.hpp"
#include "gpu/device.hpp"
#include "gpu/context.hpp"
#include "mining/miner.hpp"
#include "mining/validator.hpp"

#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

namespace {

// Signal-safe shutdown flag (only sig_atomic_t is guaranteed safe in signal handlers)
volatile std::sig_atomic_t g_shutdown_requested = 0;

void signal_handler(int) {
    g_shutdown_requested = 1;
}

} // anonymous namespace

int main() {
    using namespace shallenge;

    std::string username = config::username;

    // Compile-time config sanity check
    if (username.length() + 1 + config::nonce_len != 32) {
        std::cerr << "Username must be " << (32 - 1 - config::nonce_len)
                  << " characters (got " << username.length() << ")" << std::endl;
        return 1;
    }

    // Initialize shared state
    SharedState shared;
    shared.username = username;
    shared.best_hash = std::vector<uint32_t>(config::initial_target, config::initial_target + 8);
    shared.start_time = std::chrono::steady_clock::now();

    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "Shallenge Miner (OpenCL Multi-GPU)" << std::endl;
    std::cout << "Username: " << username << std::endl;
    std::cout << "Initial target: " << uint_to_hex(config::initial_target, 8) << std::endl;
    std::cout << "Global size: " << config::global_size << " threads per launch per GPU" << std::endl;
    std::cout << "Local size: " << config::local_size << " threads per work-group" << std::endl;

    // Discover all GPUs
    std::vector<cl_device_id> devices = discover_all_gpus();
    if (devices.empty()) {
        std::cerr << "No GPUs found!" << std::endl;
        return 1;
    }
    std::cout << "Found " << devices.size() << " GPU(s)" << std::endl;

    // Initialize all GPUs (RAII handles cleanup automatically)
    std::vector<GPUContext> gpus;
    gpus.reserve(devices.size());

    for (size_t i = 0; i < devices.size(); i++) {
        auto ctx = create_gpu_context(devices[i], static_cast<int>(i), username);
        if (!ctx) {
            std::cerr << "Failed to initialize GPU " << i << std::endl;
            return 1;
        }
        std::cout << "Initialized GPU " << i << ": " << ctx->device_name << std::endl;
        gpus.push_back(std::move(*ctx));
    }

    // Validate each GPU's SHA-256 implementation
    std::cout << "\nValidating GPU kernels..." << std::endl;
    for (auto& gpu : gpus) {
        if (!validate_gpu(gpu, username)) {
            std::cerr << "GPU validation failed - aborting" << std::endl;
            return 1;
        }
    }

    std::cout << "\nMining started...\n" << std::endl;

    // Start worker threads
    std::vector<std::thread> worker_threads;
    for (auto& gpu : gpus) {
        worker_threads.emplace_back(gpu_worker_thread, std::ref(gpu), std::ref(shared));
    }

    // Stats reporting loop in main thread
    uint64_t last_total = 0;
    auto last_time = std::chrono::steady_clock::now();

    while (shared.running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Check for signal-triggered shutdown
        if (g_shutdown_requested) {
            std::cout << "\nShutting down..." << std::endl;
            shared.running.store(false);
            break;
        }

        // Only print stats every 5 seconds worth of iterations
        static int tick = 0;
        if (++tick < 5) continue;
        tick = 0;

        uint64_t total_hashes = 0;
        uint64_t total_matches = 0;
        for (const auto& gpu : gpus) {
            total_hashes += gpu.hashes_computed.load();
            total_matches += gpu.matches_found.load();
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - shared.start_time).count();
        auto interval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();

        double recent_rate = (interval_ms > 0) ?
            static_cast<double>(total_hashes - last_total) / (interval_ms / 1000.0) : 0;

        last_total = total_hashes;
        last_time = now;

        double per_gpu_rate = recent_rate / gpus.size();

        std::cout << "\r[Stats] " << std::fixed << std::setprecision(2)
                  << (recent_rate / 1e6) << " MH/s (" << (per_gpu_rate / 1e6) << " MH/s/GPU), "
                  << std::setprecision(3) << (total_hashes / 1e9) << "B hashes, "
                  << total_matches << " matches, "
                  << elapsed << "s elapsed" << std::flush;
    }

    // Wait for all threads
    for (auto& t : worker_threads) {
        t.join();
    }

    // Print final stats
    uint64_t total_hashes = 0;
    uint64_t total_matches = 0;
    for (const auto& gpu : gpus) {
        total_hashes += gpu.hashes_computed.load();
        total_matches += gpu.matches_found.load();
    }

    std::cout << "\n\nFinal Results:" << std::endl;
    std::cout << "  Total hashes: " << total_hashes << std::endl;
    std::cout << "  Total matches: " << total_matches << std::endl;
    std::cout << "  Best hash: " << uint_to_hex(shared.best_hash.data(), 8) << std::endl;
    if (!shared.best_nonce.empty()) {
        std::cout << "  Best nonce: " << shared.best_nonce << std::endl;
        std::cout << "  Challenge: " << username << "/" << shared.best_nonce << std::endl;
    }

    // Cleanup happens automatically via RAII when gpus vector goes out of scope
    return 0;
}
