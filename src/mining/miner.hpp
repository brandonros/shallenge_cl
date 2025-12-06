#pragma once

#include "../config.hpp"
#include "../core/hash_utils.hpp"
#include "../core/types.hpp"
#include "../gpu/cl_error.hpp"
#include "../gpu/context.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <vector>

namespace shallenge {

// Worker thread function for each GPU
inline void gpu_worker_thread(GPUContext& ctx, SharedState& shared) {
    std::cout << "[GPU " << ctx.device_index << "] Started mining on " << ctx.device_name << std::endl;

    while (shared.running.load()) {
        // Get current best hash
        std::vector<uint32_t> current_target(8);
        {
            std::lock_guard<std::mutex> lock(shared.best_hash_mutex);
            current_target = shared.best_hash;
        }

        cl_uint rng_seed = static_cast<cl_uint>(ctx.rng());

        // Reset found count and update target
        cl_uint zero = 0;
        clEnqueueWriteBuffer(ctx.queue, ctx.found_count_buf, CL_FALSE, 0, sizeof(cl_uint), &zero, 0, nullptr, nullptr);
        clEnqueueWriteBuffer(ctx.queue, ctx.target_hash_buf, CL_FALSE, 0, 8 * sizeof(cl_uint), current_target.data(), 0, nullptr, nullptr);

        clSetKernelArg(ctx.kernel, 3, sizeof(cl_uint), &rng_seed);

        size_t global_size = config::global_size;
        size_t local_size = config::local_size;
        cl_int err = clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "[GPU " << ctx.device_index << "] Kernel error: " << cl_error_string(err) << std::endl;
            break;
        }

        clFinish(ctx.queue);
        ctx.hashes_computed.fetch_add(static_cast<uint64_t>(config::global_size) * config::hashes_per_thread);

        // Check for matches
        cl_uint found_count;
        clEnqueueReadBuffer(ctx.queue, ctx.found_count_buf, CL_TRUE, 0, sizeof(cl_uint), &found_count, 0, nullptr, nullptr);

        if (found_count > 0) {
            // Cap at max_results (kernel may have found more but only stored this many)
            size_t results_to_read = std::min(static_cast<size_t>(found_count), config::max_results);

            // Read all results
            std::vector<uint8_t> all_hashes(results_to_read * 32);
            std::vector<uint8_t> all_nonces(results_to_read * 32);

            clEnqueueReadBuffer(ctx.queue, ctx.found_hashes_buf, CL_TRUE, 0,
                               results_to_read * 32, all_hashes.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(ctx.queue, ctx.found_nonces_buf, CL_TRUE, 0,
                               results_to_read * 32, all_nonces.data(), 0, nullptr, nullptr);

            // Find the best result among all found
            size_t best_idx = 0;
            std::vector<uint32_t> best_hash_uint(8);
            bytes_to_uint(all_hashes.data(), best_hash_uint.data(), 8);

            for (size_t i = 1; i < results_to_read; i++) {
                std::vector<uint32_t> this_hash_uint(8);
                bytes_to_uint(all_hashes.data() + i * 32, this_hash_uint.data(), 8);

                if (compare_hashes_uint(this_hash_uint.data(), best_hash_uint.data()) < 0) {
                    best_idx = i;
                    best_hash_uint = this_hash_uint;
                }
            }

            // Try to update global best hash
            {
                std::lock_guard<std::mutex> lock(shared.best_hash_mutex);
                if (compare_hashes_uint(best_hash_uint.data(), shared.best_hash.data()) < 0) {
                    shared.best_hash = best_hash_uint;
                    shared.best_nonce = std::string(
                        reinterpret_cast<char*>(all_nonces.data() + best_idx * 32), config::nonce_len);
                    ctx.matches_found.fetch_add(1);

                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - shared.start_time).count();

                    std::cout << "\n[GPU " << ctx.device_index << "] NEW BEST FOUND!" << std::endl;
                    std::cout << "  Hash: " << bytes_to_hex(all_hashes.data() + best_idx * 32, 32) << std::endl;
                    std::cout << "  Zeroes: " << count_leading_zeros(bytes_to_hex(all_hashes.data() + best_idx * 32, 32)) << std::endl;
                    std::cout << "  Nonce: " << shared.best_nonce << std::endl;
                    std::cout << "  Challenge: " << shared.username << "/" << shared.best_nonce << std::endl;
                    std::cout << "  Time: " << elapsed << "s elapsed" << std::endl;
                    std::cout << "  (Found " << found_count << " candidates this batch)" << std::endl;
                    std::cout << std::endl;
                }
            }
        }
    }

    std::cout << "[GPU " << ctx.device_index << "] Stopped" << std::endl;
}

} // namespace shallenge
