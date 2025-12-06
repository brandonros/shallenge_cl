#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "kernel.h"

// Configuration
const char* DEFAULT_USERNAME = "brandonros";
const size_t NONCE_LEN = 21;
const size_t GLOBAL_SIZE = 1024 * 1024;  // 1M threads per kernel launch

// Convert bytes to hex string
std::string bytes_to_hex(const uint8_t* data, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; i++) {
        ss << std::setw(2) << static_cast<int>(data[i]);
    }
    return ss.str();
}

// Parse hex string to bytes
bool hex_to_bytes(const std::string& hex, uint8_t* out, size_t out_len) {
    if (hex.length() != out_len * 2) return false;
    for (size_t i = 0; i < out_len; i++) {
        unsigned int byte;
        if (sscanf(hex.c_str() + i * 2, "%2x", &byte) != 1) return false;
        out[i] = static_cast<uint8_t>(byte);
    }
    return true;
}

// Compare two 32-byte hashes lexicographically
// Returns: -1 if a < b, 0 if a == b, 1 if a > b
int compare_hashes(const uint8_t* a, const uint8_t* b) {
    for (int i = 0; i < 32; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    // Parse arguments
    std::string username = DEFAULT_USERNAME;
    std::vector<uint8_t> target_hash(32, 0xFF);  // Start with max hash (all 0xFF)

    if (argc >= 2) {
        username = argv[1];
    }
    if (argc >= 3) {
        if (!hex_to_bytes(argv[2], target_hash.data(), 32)) {
            std::cerr << "Invalid target hash format. Expected 64 hex characters." << std::endl;
            return 1;
        }
    }

    // Validate username length (username + "/" + nonce must = 32 bytes)
    if (username.length() + 1 + NONCE_LEN != 32) {
        std::cerr << "Username must be " << (32 - 1 - NONCE_LEN) << " characters (got " << username.length() << ")" << std::endl;
        return 1;
    }

    std::cout << "Shallenge Miner (OpenCL)" << std::endl;
    std::cout << "Username: " << username << std::endl;
    std::cout << "Initial target: " << bytes_to_hex(target_hash.data(), 32) << std::endl;
    std::cout << "Global size: " << GLOBAL_SIZE << " threads per launch" << std::endl;

    cl_int err;

    // Get platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get platform" << std::endl;
        return 1;
    }

    // Get device (prefer GPU)
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "No GPU found, trying CPU..." << std::endl;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to get any device" << std::endl;
            return 1;
        }
    }

    // Print device name
    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    std::cout << "Using device: " << device_name << std::endl;

    // Create context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context" << std::endl;
        return 1;
    }

    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue" << std::endl;
        return 1;
    }

    // Load and build kernel (embedded at compile time via xxd)
    const char* src = reinterpret_cast<const char*>(src_kernel_combined_cl);
    size_t srcLen = src_kernel_combined_cl_len;

    cl_program program = clCreateProgramWithSource(context, 1, &src, &srcLen, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program" << std::endl;
        return 1;
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build program" << std::endl;
        char log[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << log << std::endl;
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "shallenge_mine", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel" << std::endl;
        return 1;
    }

    // Create buffers
    cl_mem username_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          username.length(), (void*)username.c_str(), &err);
    cl_mem target_hash_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                             32, nullptr, &err);
    cl_mem found_count_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             sizeof(cl_uint), nullptr, &err);
    cl_mem found_hash_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                            32, nullptr, &err);
    cl_mem found_nonce_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                             NONCE_LEN, nullptr, &err);
    cl_mem found_thread_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                              sizeof(cl_uint), nullptr, &err);

    // Set static kernel arguments
    cl_uint username_len = static_cast<cl_uint>(username.length());
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &username_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_uint), &username_len);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &target_hash_buf);
    // arg 3 (rng_seed) set per iteration
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &found_count_buf);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &found_hash_buf);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &found_nonce_buf);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), &found_thread_buf);

    // Random number generator for seeds
    std::random_device rd;
    std::mt19937_64 rng(rd());

    // Stats tracking
    auto start_time = std::chrono::steady_clock::now();
    uint64_t total_hashes = 0;
    uint64_t total_matches = 0;

    // Best hash found
    std::vector<uint8_t> best_hash = target_hash;
    std::string best_nonce;

    std::cout << "\nMining started...\n" << std::endl;

    // Main mining loop
    while (true) {
        // Generate random seed for this batch
        cl_ulong rng_seed = rng();

        // Reset found count
        cl_uint zero = 0;
        clEnqueueWriteBuffer(queue, found_count_buf, CL_FALSE, 0, sizeof(cl_uint), &zero, 0, nullptr, nullptr);

        // Update target hash buffer
        clEnqueueWriteBuffer(queue, target_hash_buf, CL_FALSE, 0, 32, best_hash.data(), 0, nullptr, nullptr);

        // Set rng_seed argument
        clSetKernelArg(kernel, 3, sizeof(cl_ulong), &rng_seed);

        // Execute kernel
        size_t global_size = GLOBAL_SIZE;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to enqueue kernel (error " << err << ")" << std::endl;
            break;
        }

        // Wait for completion
        clFinish(queue);

        total_hashes += GLOBAL_SIZE;

        // Check if any matches were found
        cl_uint found_count;
        clEnqueueReadBuffer(queue, found_count_buf, CL_TRUE, 0, sizeof(cl_uint), &found_count, 0, nullptr, nullptr);

        if (found_count > 0) {
            // Read the found hash and nonce
            std::vector<uint8_t> found_hash(32);
            std::vector<char> found_nonce(NONCE_LEN + 1, 0);
            cl_uint found_thread_idx;

            clEnqueueReadBuffer(queue, found_hash_buf, CL_TRUE, 0, 32, found_hash.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(queue, found_nonce_buf, CL_TRUE, 0, NONCE_LEN, found_nonce.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(queue, found_thread_buf, CL_TRUE, 0, sizeof(cl_uint), &found_thread_idx, 0, nullptr, nullptr);

            // Check if this is actually better than current best
            if (compare_hashes(found_hash.data(), best_hash.data()) < 0) {
                best_hash = found_hash;
                best_nonce = std::string(found_nonce.data(), NONCE_LEN);
                total_matches++;

                // Print the new best
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                double hashes_per_sec = elapsed > 0 ? static_cast<double>(total_hashes) / elapsed : 0;

                std::cout << "NEW BEST FOUND!" << std::endl;
                std::cout << "  Hash: " << bytes_to_hex(best_hash.data(), 32) << std::endl;
                std::cout << "  Nonce: " << best_nonce << std::endl;
                std::cout << "  Challenge: " << username << "/" << best_nonce << std::endl;
                std::cout << "  Thread: " << found_thread_idx << ", Seed: " << rng_seed << std::endl;
                std::cout << "  Stats: " << total_matches << " matches, "
                          << std::fixed << std::setprecision(2) << (hashes_per_sec / 1e6) << " MH/s, "
                          << elapsed << "s elapsed" << std::endl;
                std::cout << std::endl;
            }
        }

        // Periodic stats update
        static uint64_t last_stats_hashes = 0;
        if (total_hashes - last_stats_hashes >= GLOBAL_SIZE * 100) {  // Every 100 batches
            last_stats_hashes = total_hashes;
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            double hashes_per_sec = elapsed > 0 ? static_cast<double>(total_hashes) / elapsed : 0;

            std::cout << "\rStats: " << std::fixed << std::setprecision(2) << (hashes_per_sec / 1e6)
                      << " MH/s, " << (total_hashes / 1e9) << "B hashes, "
                      << total_matches << " matches" << std::flush;
        }
    }

    // Cleanup
    clReleaseMemObject(username_buf);
    clReleaseMemObject(target_hash_buf);
    clReleaseMemObject(found_count_buf);
    clReleaseMemObject(found_hash_buf);
    clReleaseMemObject(found_nonce_buf);
    clReleaseMemObject(found_thread_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
