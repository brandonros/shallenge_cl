#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <csignal>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "kernel.h"

// Configuration
const char* DEFAULT_USERNAME = "brandonros";
const size_t USERNAME_LEN = 10;
const size_t SHA256_BLOCK_SIZE = 32;
const size_t SEPARATOR_LEN = 1;
const size_t NONCE_LEN = SHA256_BLOCK_SIZE - (USERNAME_LEN + SEPARATOR_LEN);
const size_t GLOBAL_SIZE = 1024 * 1024;  // 1M threads per kernel launch
const size_t LOCAL_SIZE = 256;           // Work-group size
const size_t HASHES_PER_THREAD = 64;     // Inner loop iterations per thread

// Per-GPU context
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
    cl_mem found_hash_buf;
    cl_mem found_nonce_buf;
    cl_mem found_thread_buf;

    std::mt19937_64 rng;
    std::atomic<uint64_t> hashes_computed{0};
    std::atomic<uint64_t> matches_found{0};
};

// Shared state across all GPUs
struct SharedState {
    std::mutex best_hash_mutex;
    std::vector<uint32_t> best_hash;
    std::string best_nonce;

    std::atomic<bool> running{true};

    std::string username;
    std::chrono::steady_clock::time_point start_time;
};

// Global pointer for signal handler
SharedState* g_shared_state = nullptr;

void signal_handler(int) {
    if (g_shared_state) {
        std::cout << "\nShutting down..." << std::endl;
        g_shared_state->running.store(false);
    }
}

// Convert bytes to hex string
std::string bytes_to_hex(const uint8_t* data, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; i++) {
        ss << std::setw(2) << static_cast<int>(data[i]);
    }
    return ss.str();
}

// Count leading zero nibbles
int count_leading_zeros(const std::string& hex_string) {
    int count = 0;
    for (char c : hex_string) {
        if (c == '0') count++;
        else break;
    }
    return count;
}

// Convert uint32_t array to hex string (big-endian)
std::string uint_to_hex(const uint32_t* data, size_t count) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < count; i++) {
        ss << std::setw(8) << data[i];
    }
    return ss.str();
}

// Parse hex string to uint32_t array (big-endian)
bool hex_to_uint(const std::string& hex, uint32_t* out, size_t out_count) {
    if (hex.length() != out_count * 8) return false;
    for (size_t i = 0; i < out_count; i++) {
        unsigned int word;
        if (sscanf(hex.c_str() + i * 8, "%8x", &word) != 1) return false;
        out[i] = static_cast<uint32_t>(word);
    }
    return true;
}

// Convert bytes to uint32_t array (big-endian)
void bytes_to_uint(const uint8_t* bytes, uint32_t* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = (static_cast<uint32_t>(bytes[i*4]) << 24) |
                 (static_cast<uint32_t>(bytes[i*4+1]) << 16) |
                 (static_cast<uint32_t>(bytes[i*4+2]) << 8) |
                 static_cast<uint32_t>(bytes[i*4+3]);
    }
}

// Compare two 8-word hashes lexicographically
int compare_hashes_uint(const uint32_t* a, const uint32_t* b) {
    for (int i = 0; i < 8; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

// Discover all available GPUs across all platforms
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

// Initialize a single GPU context
bool initialize_gpu(GPUContext& ctx, cl_device_id device, int device_index, const std::string& username) {
    ctx.device_index = device_index;
    ctx.device = device;

    char name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    ctx.device_name = name;

    cl_int err;

    ctx.context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return false;

#ifdef CL_VERSION_2_0
    ctx.queue = clCreateCommandQueueWithProperties(ctx.context, device, nullptr, &err);
#else
    ctx.queue = clCreateCommandQueue(ctx.context, device, 0, &err);
#endif
    if (err != CL_SUCCESS) return false;

    const char* src = reinterpret_cast<const char*>(output_kernel_combined_cl);
    size_t srcLen = output_kernel_combined_cl_len;

    ctx.program = clCreateProgramWithSource(ctx.context, 1, &src, &srcLen, &err);
    if (err != CL_SUCCESS) return false;

    err = clBuildProgram(ctx.program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(ctx.program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "[GPU " << device_index << "] Build error: " << log << std::endl;
        return false;
    }

    ctx.kernel = clCreateKernel(ctx.program, "shallenge_mine", &err);
    if (err != CL_SUCCESS) return false;

    ctx.username_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       username.length(), (void*)username.c_str(), &err);
    ctx.target_hash_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY,
                                          8 * sizeof(cl_uint), nullptr, &err);
    ctx.found_count_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE,
                                          sizeof(cl_uint), nullptr, &err);
    ctx.found_hash_buf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, 32, nullptr, &err);
    ctx.found_nonce_buf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, NONCE_LEN, nullptr, &err);
    ctx.found_thread_buf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, sizeof(cl_uint), nullptr, &err);

    cl_uint username_len = static_cast<cl_uint>(username.length());
    clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &ctx.username_buf);
    clSetKernelArg(ctx.kernel, 1, sizeof(cl_uint), &username_len);
    clSetKernelArg(ctx.kernel, 2, sizeof(cl_mem), &ctx.target_hash_buf);
    clSetKernelArg(ctx.kernel, 4, sizeof(cl_mem), &ctx.found_count_buf);
    clSetKernelArg(ctx.kernel, 5, sizeof(cl_mem), &ctx.found_hash_buf);
    clSetKernelArg(ctx.kernel, 6, sizeof(cl_mem), &ctx.found_nonce_buf);
    clSetKernelArg(ctx.kernel, 7, sizeof(cl_mem), &ctx.found_thread_buf);
    clSetKernelArg(ctx.kernel, 8, sizeof(cl_uint) * 8, nullptr);

    std::random_device rd;
    ctx.rng.seed(rd() + static_cast<uint64_t>(device_index) * 0x9E3779B97F4A7C15ULL);

    return true;
}

// Cleanup GPU resources
void cleanup_gpu(GPUContext& ctx) {
    if (ctx.username_buf) clReleaseMemObject(ctx.username_buf);
    if (ctx.target_hash_buf) clReleaseMemObject(ctx.target_hash_buf);
    if (ctx.found_count_buf) clReleaseMemObject(ctx.found_count_buf);
    if (ctx.found_hash_buf) clReleaseMemObject(ctx.found_hash_buf);
    if (ctx.found_nonce_buf) clReleaseMemObject(ctx.found_nonce_buf);
    if (ctx.found_thread_buf) clReleaseMemObject(ctx.found_thread_buf);
    if (ctx.kernel) clReleaseKernel(ctx.kernel);
    if (ctx.program) clReleaseProgram(ctx.program);
    if (ctx.queue) clReleaseCommandQueue(ctx.queue);
    if (ctx.context) clReleaseContext(ctx.context);
}

// Worker thread function for each GPU
void gpu_worker_thread(GPUContext& ctx, SharedState& shared) {
    std::cout << "[GPU " << ctx.device_index << "] Started mining on " << ctx.device_name << std::endl;

    while (shared.running.load()) {
        // Get current best hash
        std::vector<uint32_t> current_target(8);
        {
            std::lock_guard<std::mutex> lock(shared.best_hash_mutex);
            current_target = shared.best_hash;
        }

        cl_ulong rng_seed = ctx.rng();

        // Reset found count and update target
        cl_uint zero = 0;
        clEnqueueWriteBuffer(ctx.queue, ctx.found_count_buf, CL_FALSE, 0, sizeof(cl_uint), &zero, 0, nullptr, nullptr);
        clEnqueueWriteBuffer(ctx.queue, ctx.target_hash_buf, CL_FALSE, 0, 8 * sizeof(cl_uint), current_target.data(), 0, nullptr, nullptr);

        clSetKernelArg(ctx.kernel, 3, sizeof(cl_ulong), &rng_seed);

        size_t global_size = GLOBAL_SIZE;
        size_t local_size = LOCAL_SIZE;
        cl_int err = clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "[GPU " << ctx.device_index << "] Kernel error: " << err << std::endl;
            break;
        }

        clFinish(ctx.queue);
        ctx.hashes_computed.fetch_add(GLOBAL_SIZE * HASHES_PER_THREAD);

        // Check for matches
        cl_uint found_count;
        clEnqueueReadBuffer(ctx.queue, ctx.found_count_buf, CL_TRUE, 0, sizeof(cl_uint), &found_count, 0, nullptr, nullptr);

        if (found_count > 0) {
            std::vector<uint8_t> found_hash(32);
            std::vector<char> found_nonce(NONCE_LEN + 1, 0);
            cl_uint found_thread_idx;

            clEnqueueReadBuffer(ctx.queue, ctx.found_hash_buf, CL_TRUE, 0, 32, found_hash.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(ctx.queue, ctx.found_nonce_buf, CL_TRUE, 0, NONCE_LEN, found_nonce.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(ctx.queue, ctx.found_thread_buf, CL_TRUE, 0, sizeof(cl_uint), &found_thread_idx, 0, nullptr, nullptr);

            std::vector<uint32_t> found_hash_uint(8);
            bytes_to_uint(found_hash.data(), found_hash_uint.data(), 8);

            // Try to update best hash
            {
                std::lock_guard<std::mutex> lock(shared.best_hash_mutex);
                if (compare_hashes_uint(found_hash_uint.data(), shared.best_hash.data()) < 0) {
                    shared.best_hash = found_hash_uint;
                    shared.best_nonce = std::string(found_nonce.data(), NONCE_LEN);
                    ctx.matches_found.fetch_add(1);

                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - shared.start_time).count();

                    std::cout << "\n[GPU " << ctx.device_index << "] NEW BEST FOUND!" << std::endl;
                    std::cout << "  Hash: " << bytes_to_hex(found_hash.data(), 32) << std::endl;
                    std::cout << "  Zeroes: " << count_leading_zeros(bytes_to_hex(found_hash.data(), 32)) << std::endl;
                    std::cout << "  Nonce: " << shared.best_nonce << std::endl;
                    std::cout << "  Challenge: " << shared.username << "/" << shared.best_nonce << std::endl;
                    std::cout << "  Time: " << elapsed << "s elapsed" << std::endl;
                    std::cout << std::endl;
                }
            }
        }
    }

    std::cout << "[GPU " << ctx.device_index << "] Stopped" << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse arguments
    std::string username = DEFAULT_USERNAME;
    std::vector<uint32_t> initial_target(8, 0xFFFFFFFF);

    if (argc >= 2) {
        username = argv[1];
    }
    if (argc >= 3) {
        if (!hex_to_uint(argv[2], initial_target.data(), 8)) {
            std::cerr << "Invalid target hash format. Expected 64 hex characters." << std::endl;
            return 1;
        }
    }

    // Validate username length
    if (username.length() + 1 + NONCE_LEN != 32) {
        std::cerr << "Username must be " << (32 - 1 - NONCE_LEN) << " characters (got " << username.length() << ")" << std::endl;
        return 1;
    }

    // Initialize shared state
    SharedState shared;
    shared.username = username;
    shared.best_hash = initial_target;
    shared.start_time = std::chrono::steady_clock::now();
    g_shared_state = &shared;

    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    std::cout << "Shallenge Miner (OpenCL Multi-GPU)" << std::endl;
    std::cout << "Username: " << username << std::endl;
    std::cout << "Initial target: " << uint_to_hex(initial_target.data(), 8) << std::endl;
    std::cout << "Global size: " << GLOBAL_SIZE << " threads per launch per GPU" << std::endl;
    std::cout << "Local size: " << LOCAL_SIZE << " threads per work-group" << std::endl;

    // Discover all GPUs
    std::vector<cl_device_id> devices = discover_all_gpus();
    if (devices.empty()) {
        std::cerr << "No GPUs found!" << std::endl;
        return 1;
    }
    std::cout << "Found " << devices.size() << " GPU(s)" << std::endl;

    // Initialize all GPUs
    std::vector<GPUContext> gpus(devices.size());
    for (size_t i = 0; i < devices.size(); i++) {
        if (!initialize_gpu(gpus[i], devices[i], static_cast<int>(i), username)) {
            std::cerr << "Failed to initialize GPU " << i << std::endl;
            for (size_t j = 0; j < i; j++) {
                cleanup_gpu(gpus[j]);
            }
            return 1;
        }
        std::cout << "Initialized GPU " << i << ": " << gpus[i].device_name << std::endl;
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
        std::this_thread::sleep_for(std::chrono::seconds(5));

        if (!shared.running.load()) break;

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

    // Cleanup
    for (auto& gpu : gpus) {
        cleanup_gpu(gpu);
    }

    return 0;
}
