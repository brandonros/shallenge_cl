// Shallenge OpenCL Miner - Consolidated
// SHA-256 mining for https://shallenge.quirino.net/

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "kernel.h"

// ============================================================================
// Configuration (from Makefile -D flags)
// ============================================================================

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)

constexpr const char* USERNAME = DEFAULT_USERNAME;
constexpr size_t USERNAME_LEN = sizeof(DEFAULT_USERNAME) - 1;
constexpr size_t NONCE_LEN = 32 - USERNAME_LEN - 1;
constexpr size_t MAX_RESULTS = 64;

// 8 leading zero nibbles
constexpr uint32_t INITIAL_TARGET[8] = {
    0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// ============================================================================
// Hash Utilities
// ============================================================================

std::string bytes_to_hex(const uint8_t* data, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; i++) {
        ss << std::setw(2) << static_cast<int>(data[i]);
    }
    return ss.str();
}

int count_leading_zeros(const std::string& hex_string) {
    int count = 0;
    for (char c : hex_string) {
        if (c == '0') count++;
        else break;
    }
    return count;
}

std::string uint_to_hex(const uint32_t* data, size_t count) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < count; i++) {
        ss << std::setw(8) << data[i];
    }
    return ss.str();
}

void bytes_to_uint(const uint8_t* bytes, uint32_t* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = (static_cast<uint32_t>(bytes[i*4]) << 24) |
                 (static_cast<uint32_t>(bytes[i*4+1]) << 16) |
                 (static_cast<uint32_t>(bytes[i*4+2]) << 8) |
                 static_cast<uint32_t>(bytes[i*4+3]);
    }
}

int compare_hashes_uint(const uint32_t* a, const uint32_t* b) {
    for (int i = 0; i < 8; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

// ============================================================================
// Shared State
// ============================================================================

struct SharedState {
    std::mutex best_hash_mutex;
    std::vector<uint32_t> best_hash;
    std::string best_nonce;
    std::atomic<bool> running{true};
    std::string username;
    std::chrono::steady_clock::time_point start_time;
};

// ============================================================================
// GPU Context
// ============================================================================

struct GPUContext {
    int device_index;
    std::string device_name;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem username_buf;
    cl_mem target_hash_buf;
    cl_mem found_count_buf;
    cl_mem found_hashes_buf;
    cl_mem found_nonces_buf;
    cl_mem found_thread_ids_buf;
    std::mt19937_64 rng;
    std::atomic<uint64_t> hashes_computed{0};
    std::atomic<uint64_t> matches_found{0};
};

// ============================================================================
// GPU Discovery
// ============================================================================

std::vector<cl_device_id> discover_all_gpus() {
    std::vector<cl_device_id> all_devices;

    cl_uint num_platforms;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        return all_devices;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    for (cl_platform_id platform : platforms) {
        cl_uint num_devices;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) == CL_SUCCESS && num_devices > 0) {
            std::vector<cl_device_id> devices(num_devices);
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
            all_devices.insert(all_devices.end(), devices.begin(), devices.end());
        }
    }

    return all_devices;
}

// ============================================================================
// GPU Context Creation
// ============================================================================

bool create_gpu_context(cl_device_id device, int device_index, const std::string& username, GPUContext& ctx) {
    ctx.device_index = device_index;
    ctx.device = device;

    char name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    ctx.device_name = name;

    cl_int err;

    ctx.context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create context: " << err << std::endl;
        return false;
    }

#ifdef CL_VERSION_2_0
    ctx.queue = clCreateCommandQueueWithProperties(ctx.context, device, nullptr, &err);
#else
    ctx.queue = clCreateCommandQueue(ctx.context, device, 0, &err);
#endif
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create command queue: " << err << std::endl;
        return false;
    }

    const char* src = reinterpret_cast<const char*>(src_shallenge_cl);
    size_t srcLen = src_shallenge_cl_len;

    ctx.program = clCreateProgramWithSource(ctx.context, 1, &src, &srcLen, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create program: " << err << std::endl;
        return false;
    }

    const char* buildOpts = "-D HASHES_PER_THREAD=" STRINGIFY(HASHES_PER_THREAD);
    err = clBuildProgram(ctx.program, 1, &device, buildOpts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(ctx.program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "[GPU " << device_index << "] Build error: " << log << std::endl;
        return false;
    }

    ctx.kernel = clCreateKernel(ctx.program, "shallenge_mine", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create kernel: " << err << std::endl;
        return false;
    }

    ctx.username_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       username.length(), (void*)username.c_str(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create username buffer: " << err << std::endl;
        return false;
    }

    ctx.target_hash_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, 8 * sizeof(cl_uint), nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create target hash buffer: " << err << std::endl;
        return false;
    }

    ctx.found_count_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create found count buffer: " << err << std::endl;
        return false;
    }

    ctx.found_hashes_buf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, MAX_RESULTS * 32, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create found hashes buffer: " << err << std::endl;
        return false;
    }

    ctx.found_nonces_buf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, MAX_RESULTS * 32, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create found nonces buffer: " << err << std::endl;
        return false;
    }

    ctx.found_thread_ids_buf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, MAX_RESULTS * sizeof(cl_uint), nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create found thread IDs buffer: " << err << std::endl;
        return false;
    }

    cl_uint username_len = static_cast<cl_uint>(username.length());
    clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &ctx.username_buf);
    clSetKernelArg(ctx.kernel, 1, sizeof(cl_uint), &username_len);
    clSetKernelArg(ctx.kernel, 2, sizeof(cl_mem), &ctx.target_hash_buf);
    // arg 3 (rng_seed_lo) set per launch
    // arg 4 (rng_seed_hi) set per launch
    clSetKernelArg(ctx.kernel, 5, sizeof(cl_mem), &ctx.found_count_buf);
    clSetKernelArg(ctx.kernel, 6, sizeof(cl_mem), &ctx.found_hashes_buf);
    clSetKernelArg(ctx.kernel, 7, sizeof(cl_mem), &ctx.found_nonces_buf);
    clSetKernelArg(ctx.kernel, 8, sizeof(cl_mem), &ctx.found_thread_ids_buf);
    clSetKernelArg(ctx.kernel, 9, sizeof(cl_uint) * 8, nullptr);

    std::random_device rd;
    ctx.rng.seed(rd() + static_cast<uint64_t>(device_index) * 0x9E3779B97F4A7C15ULL);

    return true;
}

// ============================================================================
// Mining Worker Thread
// ============================================================================

void gpu_worker_thread(GPUContext& ctx, SharedState& shared) {
    std::cout << "[GPU " << ctx.device_index << "] Started mining on " << ctx.device_name << std::endl;

    while (shared.running.load()) {
        std::vector<uint32_t> current_target(8);
        {
            std::lock_guard<std::mutex> lock(shared.best_hash_mutex);
            current_target = shared.best_hash;
        }

        cl_uint rng_seed_lo = static_cast<cl_uint>(ctx.rng());
        cl_uint rng_seed_hi = static_cast<cl_uint>(ctx.rng());

        cl_uint zero = 0;
        clEnqueueWriteBuffer(ctx.queue, ctx.found_count_buf, CL_FALSE, 0, sizeof(cl_uint), &zero, 0, nullptr, nullptr);
        clEnqueueWriteBuffer(ctx.queue, ctx.target_hash_buf, CL_FALSE, 0, 8 * sizeof(cl_uint), current_target.data(), 0, nullptr, nullptr);

        clSetKernelArg(ctx.kernel, 3, sizeof(cl_uint), &rng_seed_lo);
        clSetKernelArg(ctx.kernel, 4, sizeof(cl_uint), &rng_seed_hi);

        size_t global_size = GLOBAL_SIZE;
        size_t local_size = LOCAL_SIZE;
        cl_int err = clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "[GPU " << ctx.device_index << "] Kernel error: " << err << std::endl;
            break;
        }

        clFinish(ctx.queue);
        ctx.hashes_computed.fetch_add(static_cast<uint64_t>(GLOBAL_SIZE) * HASHES_PER_THREAD);

        cl_uint found_count;
        clEnqueueReadBuffer(ctx.queue, ctx.found_count_buf, CL_TRUE, 0, sizeof(cl_uint), &found_count, 0, nullptr, nullptr);

        if (found_count > 0) {
            size_t results_to_read = std::min(static_cast<size_t>(found_count), MAX_RESULTS);

            std::vector<uint8_t> all_hashes(results_to_read * 32);
            std::vector<uint8_t> all_nonces(results_to_read * 32);
            std::vector<cl_uint> all_thread_ids(results_to_read);

            clEnqueueReadBuffer(ctx.queue, ctx.found_hashes_buf, CL_TRUE, 0,
                               results_to_read * 32, all_hashes.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(ctx.queue, ctx.found_nonces_buf, CL_TRUE, 0,
                               results_to_read * 32, all_nonces.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(ctx.queue, ctx.found_thread_ids_buf, CL_TRUE, 0,
                               results_to_read * sizeof(cl_uint), all_thread_ids.data(), 0, nullptr, nullptr);

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

            {
                std::lock_guard<std::mutex> lock(shared.best_hash_mutex);
                if (compare_hashes_uint(best_hash_uint.data(), shared.best_hash.data()) < 0) {
                    shared.best_hash = best_hash_uint;
                    shared.best_nonce = std::string(
                        reinterpret_cast<char*>(all_nonces.data() + best_idx * 32), NONCE_LEN);
                    ctx.matches_found.fetch_add(1);

                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - shared.start_time).count();

                    std::cout << "\n[GPU " << ctx.device_index << "] NEW BEST FOUND!" << std::endl;
                    std::cout << "  Hash: " << bytes_to_hex(all_hashes.data() + best_idx * 32, 32) << std::endl;
                    std::cout << "  Zeroes: " << count_leading_zeros(bytes_to_hex(all_hashes.data() + best_idx * 32, 32)) << std::endl;
                    std::cout << "  Nonce: " << shared.best_nonce << std::endl;
                    std::cout << "  Challenge: " << shared.username << "/" << shared.best_nonce << std::endl;
                    std::cout << "  Seed: 0x" << std::hex << rng_seed_hi << rng_seed_lo << std::dec
                              << ", ThreadIdx: " << all_thread_ids[best_idx] << std::endl;
                    std::cout << "  Time: " << elapsed << "s elapsed" << std::endl;
                    std::cout << "  (Found " << found_count << " candidates this batch)" << std::endl;
                    std::cout << std::endl;
                }
            }
        }
    }

    std::cout << "[GPU " << ctx.device_index << "] Stopped" << std::endl;
}

// ============================================================================
// Signal Handling
// ============================================================================

volatile std::sig_atomic_t g_shutdown_requested = 0;

void signal_handler(int) {
    g_shutdown_requested = 1;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::string username = USERNAME;

    if (username.length() + 1 + NONCE_LEN != 32) {
        std::cerr << "Username must be " << (32 - 1 - NONCE_LEN)
                  << " characters (got " << username.length() << ")" << std::endl;
        return 1;
    }

    SharedState shared;
    shared.username = username;
    shared.best_hash = std::vector<uint32_t>(INITIAL_TARGET, INITIAL_TARGET + 8);
    shared.start_time = std::chrono::steady_clock::now();

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "Shallenge Miner (OpenCL Multi-GPU)" << std::endl;
    std::cout << "Username: " << username << std::endl;
    std::cout << "Initial target: " << uint_to_hex(INITIAL_TARGET, 8) << std::endl;
    std::cout << "Global size: " << GLOBAL_SIZE << " threads per launch per GPU" << std::endl;
    std::cout << "Local size: " << LOCAL_SIZE << " threads per work-group" << std::endl;

    std::vector<cl_device_id> devices = discover_all_gpus();
    if (devices.empty()) {
        std::cerr << "No GPUs found!" << std::endl;
        return 1;
    }
    std::cout << "Found " << devices.size() << " GPU(s)" << std::endl;

    std::vector<GPUContext> gpus(devices.size());
    for (size_t i = 0; i < devices.size(); i++) {
        if (!create_gpu_context(devices[i], static_cast<int>(i), username, gpus[i])) {
            std::cerr << "Failed to initialize GPU " << i << std::endl;
            return 1;
        }
        std::cout << "Initialized GPU " << i << ": " << gpus[i].device_name << std::endl;
    }

    std::cout << "\nMining started...\n" << std::endl;

    std::vector<std::thread> worker_threads;
    for (auto& gpu : gpus) {
        worker_threads.emplace_back(gpu_worker_thread, std::ref(gpu), std::ref(shared));
    }

    uint64_t last_total = 0;
    auto last_time = std::chrono::steady_clock::now();

    while (shared.running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        if (g_shutdown_requested) {
            std::cout << "\nShutting down..." << std::endl;
            shared.running.store(false);
            break;
        }

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

    for (auto& t : worker_threads) {
        t.join();
    }

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

    return 0;
}
