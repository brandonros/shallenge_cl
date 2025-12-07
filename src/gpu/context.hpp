#pragma once

#include "cl_error.hpp"
#include "cl_wrappers.hpp"
#include "../config.hpp"

#include <atomic>
#include <iostream>
#include <optional>
#include <random>
#include <string>

#include "kernel.h"

namespace shallenge {

// Stringify macro for passing defines to kernel
#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)

// Per-GPU context with RAII-managed resources
struct GPUContext {
    int device_index;
    std::string device_name;
    cl_device_id device;  // Not owned - comes from OpenCL runtime

    CLContext context;
    CLCommandQueue queue;
    CLProgram program;
    CLKernel kernel;

    CLBuffer username_buf;
    CLBuffer target_hash_buf;
    CLBuffer found_count_buf;
    CLBuffer found_hashes_buf;
    CLBuffer found_nonces_buf;
    CLBuffer found_thread_ids_buf;

    std::mt19937_64 rng;
    std::atomic<uint64_t> hashes_computed{0};
    std::atomic<uint64_t> matches_found{0};

    // Default constructor
    GPUContext() = default;

    // Non-copyable
    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;

    // Custom move constructor (std::atomic is not movable, so we copy values)
    GPUContext(GPUContext&& other) noexcept
        : device_index(other.device_index)
        , device_name(std::move(other.device_name))
        , device(other.device)
        , context(std::move(other.context))
        , queue(std::move(other.queue))
        , program(std::move(other.program))
        , kernel(std::move(other.kernel))
        , username_buf(std::move(other.username_buf))
        , target_hash_buf(std::move(other.target_hash_buf))
        , found_count_buf(std::move(other.found_count_buf))
        , found_hashes_buf(std::move(other.found_hashes_buf))
        , found_nonces_buf(std::move(other.found_nonces_buf))
        , found_thread_ids_buf(std::move(other.found_thread_ids_buf))
        , rng(std::move(other.rng))
        , hashes_computed(other.hashes_computed.load())
        , matches_found(other.matches_found.load())
    {
        other.device = nullptr;
    }

    GPUContext& operator=(GPUContext&& other) noexcept {
        if (this != &other) {
            device_index = other.device_index;
            device_name = std::move(other.device_name);
            device = other.device;
            context = std::move(other.context);
            queue = std::move(other.queue);
            program = std::move(other.program);
            kernel = std::move(other.kernel);
            username_buf = std::move(other.username_buf);
            target_hash_buf = std::move(other.target_hash_buf);
            found_count_buf = std::move(other.found_count_buf);
            found_hashes_buf = std::move(other.found_hashes_buf);
            found_nonces_buf = std::move(other.found_nonces_buf);
            found_thread_ids_buf = std::move(other.found_thread_ids_buf);
            rng = std::move(other.rng);
            hashes_computed.store(other.hashes_computed.load());
            matches_found.store(other.matches_found.load());
            other.device = nullptr;
        }
        return *this;
    }
};

// Factory function to create an initialized GPU context
// Returns std::nullopt on failure, logs errors to stderr
[[nodiscard]] inline std::optional<GPUContext> create_gpu_context(
    cl_device_id device, int device_index, const std::string& username) {

    GPUContext ctx;
    ctx.device_index = device_index;
    ctx.device = device;

    char name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    ctx.device_name = name;

    cl_int err;

    // Create context
    cl_context raw_context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create context: " << cl_error_string(err) << std::endl;
        return std::nullopt;
    }
    ctx.context = CLContext(raw_context);

    // Create command queue
#ifdef CL_VERSION_2_0
    cl_command_queue raw_queue = clCreateCommandQueueWithProperties(ctx.context, device, nullptr, &err);
#else
    cl_command_queue raw_queue = clCreateCommandQueue(ctx.context, device, 0, &err);
#endif
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create command queue: " << cl_error_string(err) << std::endl;
        return std::nullopt;
    }
    ctx.queue = CLCommandQueue(raw_queue);

    // Create program from embedded kernel source
    const char* src = reinterpret_cast<const char*>(output_kernel_combined_cl);
    size_t srcLen = output_kernel_combined_cl_len;

    cl_program raw_program = clCreateProgramWithSource(ctx.context, 1, &src, &srcLen, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create program: " << cl_error_string(err) << std::endl;
        return std::nullopt;
    }
    ctx.program = CLProgram(raw_program);

    // Build program
    const char* buildOpts = "-D HASHES_PER_THREAD=" STRINGIFY(HASHES_PER_THREAD);
    err = clBuildProgram(ctx.program, 1, &device, buildOpts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(ctx.program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "[GPU " << device_index << "] Build error: " << log << std::endl;
        return std::nullopt;
    }

    // Create kernel
    cl_kernel raw_kernel = clCreateKernel(ctx.program, "shallenge_mine", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create kernel: " << cl_error_string(err) << std::endl;
        return std::nullopt;
    }
    ctx.kernel = CLKernel(raw_kernel);

    // Create buffers
    ctx.username_buf = CLBuffer(clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                username.length(), (void*)username.c_str(), &err));
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create username buffer: " << cl_error_string(err) << std::endl;
        return std::nullopt;
    }

    ctx.target_hash_buf = CLBuffer(clCreateBuffer(ctx.context, CL_MEM_READ_ONLY,
                                                   8 * sizeof(cl_uint), nullptr, &err));
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create target hash buffer: " << cl_error_string(err) << std::endl;
        return std::nullopt;
    }

    ctx.found_count_buf = CLBuffer(clCreateBuffer(ctx.context, CL_MEM_READ_WRITE,
                                                   sizeof(cl_uint), nullptr, &err));
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create found count buffer: " << cl_error_string(err) << std::endl;
        return std::nullopt;
    }

    ctx.found_hashes_buf = CLBuffer(clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY,
                                                    config::max_results * 32, nullptr, &err));
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create found hashes buffer: " << cl_error_string(err) << std::endl;
        return std::nullopt;
    }

    ctx.found_nonces_buf = CLBuffer(clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY,
                                                    config::max_results * 32, nullptr, &err));
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create found nonces buffer: " << cl_error_string(err) << std::endl;
        return std::nullopt;
    }

    ctx.found_thread_ids_buf = CLBuffer(clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY,
                                                        config::max_results * sizeof(cl_uint), nullptr, &err));
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create found thread IDs buffer: " << cl_error_string(err) << std::endl;
        return std::nullopt;
    }

    // Set kernel arguments (constant ones)
    cl_mem username_mem = ctx.username_buf;
    cl_mem target_mem = ctx.target_hash_buf;
    cl_mem count_mem = ctx.found_count_buf;
    cl_mem hashes_mem = ctx.found_hashes_buf;
    cl_mem nonces_mem = ctx.found_nonces_buf;
    cl_mem thread_ids_mem = ctx.found_thread_ids_buf;

    cl_uint username_len = static_cast<cl_uint>(username.length());
    clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &username_mem);
    clSetKernelArg(ctx.kernel, 1, sizeof(cl_uint), &username_len);
    clSetKernelArg(ctx.kernel, 2, sizeof(cl_mem), &target_mem);
    // arg 3 (rng_seed_lo) set per launch
    // arg 4 (rng_seed_hi) set per launch
    clSetKernelArg(ctx.kernel, 5, sizeof(cl_mem), &count_mem);
    clSetKernelArg(ctx.kernel, 6, sizeof(cl_mem), &hashes_mem);
    clSetKernelArg(ctx.kernel, 7, sizeof(cl_mem), &nonces_mem);
    clSetKernelArg(ctx.kernel, 8, sizeof(cl_mem), &thread_ids_mem);
    clSetKernelArg(ctx.kernel, 9, sizeof(cl_uint) * 8, nullptr);  // local memory for target

    // Seed RNG
    std::random_device rd;
    ctx.rng.seed(rd() + static_cast<uint64_t>(device_index) * 0x9E3779B97F4A7C15ULL);

    return ctx;
}

} // namespace shallenge
