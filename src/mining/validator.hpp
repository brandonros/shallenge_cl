#pragma once

#include "../config.hpp"
#include "../core/hash_utils.hpp"
#include "../gpu/cl_error.hpp"
#include "../gpu/context.hpp"

#include <iostream>
#include <vector>

namespace shallenge {

// Validate GPU kernel produces correct SHA-256 output
// Uses fixed seeds to get deterministic nonce, then checks hash matches expected
[[nodiscard]] inline bool validate_gpu(GPUContext& ctx, const std::string& username) {
    // Expected hash for DEFAULT_USERNAME with seed_lo=0x12345678, seed_hi=0x87654321, thread 0
    const char* expected_hash = "ce91f7b53a42205289d1438afcd3c302c7d1f658099a70d81286676e60d4417b";

    // Permissive target - everything matches
    std::vector<uint32_t> permissive_target(8, 0xFFFFFFFF);

    // Fixed seeds for deterministic nonce generation (64-bit entropy)
    cl_uint validation_seed_lo = 0x12345678;
    cl_uint validation_seed_hi = 0x87654321;
    cl_uint zero = 0;

    clEnqueueWriteBuffer(ctx.queue, ctx.target_hash_buf, CL_FALSE, 0,
                         8 * sizeof(cl_uint), permissive_target.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ctx.queue, ctx.found_count_buf, CL_FALSE, 0,
                         sizeof(cl_uint), &zero, 0, nullptr, nullptr);
    clSetKernelArg(ctx.kernel, 3, sizeof(cl_uint), &validation_seed_lo);
    clSetKernelArg(ctx.kernel, 4, sizeof(cl_uint), &validation_seed_hi);

    // Run single work item
    size_t global_size = 1;
    size_t local_size = 1;
    cl_int err = clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1, nullptr,
                                         &global_size, &local_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << ctx.device_index << "] Validation kernel failed: " << cl_error_string(err) << std::endl;
        return false;
    }
    clFinish(ctx.queue);

    // Read result
    cl_uint found_count;
    clEnqueueReadBuffer(ctx.queue, ctx.found_count_buf, CL_TRUE, 0,
                        sizeof(cl_uint), &found_count, 0, nullptr, nullptr);

    if (found_count == 0) {
        std::cerr << "[GPU " << ctx.device_index << "] Validation failed: no hash produced" << std::endl;
        return false;
    }

    std::vector<uint8_t> hash(32);
    std::vector<uint8_t> nonce(32);
    clEnqueueReadBuffer(ctx.queue, ctx.found_hashes_buf, CL_TRUE, 0, 32, hash.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(ctx.queue, ctx.found_nonces_buf, CL_TRUE, 0, 32, nonce.data(), 0, nullptr, nullptr);

    std::string hash_hex = bytes_to_hex(hash.data(), 32);
    std::string nonce_str(reinterpret_cast<char*>(nonce.data()), config::nonce_len);

    std::cout << "[GPU " << ctx.device_index << "] Validation (seed=0x" << std::hex << validation_seed_hi << validation_seed_lo << std::dec << "): "
              << username << "/" << nonce_str << " -> " << hash_hex << std::endl;

    if (hash_hex != expected_hash) {
        std::cerr << "[GPU " << ctx.device_index << "] SHA-256 VALIDATION FAILED! Expected: " << expected_hash << std::endl;
        return false;
    }
    return true;
}

} // namespace shallenge
