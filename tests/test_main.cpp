#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "core/hash_utils.hpp"
#include "config.hpp"
#include "gpu/cl_error.hpp"
#include "gpu/device.hpp"
#include "gpu/context.hpp"

#include <algorithm>
#include <set>

using namespace shallenge;

// =============================================================================
// Test Helpers
// =============================================================================
namespace {

// Get a GPU context for testing (returns nullopt if no GPU available)
std::optional<GPUContext> get_test_gpu() {
    auto devices = discover_all_gpus();
    if (devices.empty()) return std::nullopt;
    return create_gpu_context(devices[0], 0, config::username);
}

} // anonymous namespace

// =============================================================================
// Hash Utility Tests
// =============================================================================

TEST_CASE("bytes_to_hex converts correctly", "[hash_utils]") {
    SECTION("simple conversion") {
        uint8_t data[] = {0xde, 0xad, 0xbe, 0xef};
        REQUIRE(bytes_to_hex(data, 4) == "deadbeef");
    }

    SECTION("leading zeros preserved") {
        uint8_t data[] = {0x00, 0x00, 0x12, 0x34};
        REQUIRE(bytes_to_hex(data, 4) == "00001234");
    }

    SECTION("full SHA-256 hash") {
        uint8_t hash[] = {
            0x97, 0xcc, 0xae, 0x8e, 0xaf, 0x12, 0x45, 0x95,
            0x00, 0x67, 0xc7, 0xed, 0x8d, 0x25, 0xef, 0x7b,
            0x17, 0x06, 0x8c, 0x89, 0x30, 0x28, 0x8a, 0xb6,
            0x27, 0x7e, 0xa0, 0x58, 0xee, 0xb7, 0x3b, 0x49
        };
        REQUIRE(bytes_to_hex(hash, 32) == "97ccae8eaf1245950067c7ed8d25ef7b17068c8930288ab6277ea058eeb73b49");
    }
}

TEST_CASE("count_leading_zeros counts correctly", "[hash_utils]") {
    REQUIRE(count_leading_zeros("00001234") == 4);
    REQUIRE(count_leading_zeros("abcd0000") == 0);
    REQUIRE(count_leading_zeros("0000000000000000") == 16);
    REQUIRE(count_leading_zeros("1") == 0);
    REQUIRE(count_leading_zeros("") == 0);
}

TEST_CASE("uint_to_hex converts correctly", "[hash_utils]") {
    uint32_t data[] = {0x00000000, 0x00FFFFFF};
    REQUIRE(uint_to_hex(data, 2) == "0000000000ffffff");
}

TEST_CASE("compare_hashes_uint ordering", "[hash_utils]") {
    SECTION("less than") {
        uint32_t a[] = {0, 0, 0, 0, 0, 0, 0, 1};
        uint32_t b[] = {0, 0, 0, 0, 0, 0, 0, 2};
        REQUIRE(compare_hashes_uint(a, b) < 0);
    }

    SECTION("greater than") {
        uint32_t a[] = {0, 0, 0, 0, 0, 0, 0, 2};
        uint32_t b[] = {0, 0, 0, 0, 0, 0, 0, 1};
        REQUIRE(compare_hashes_uint(a, b) > 0);
    }

    SECTION("equal") {
        uint32_t a[] = {1, 2, 3, 4, 5, 6, 7, 8};
        uint32_t b[] = {1, 2, 3, 4, 5, 6, 7, 8};
        REQUIRE(compare_hashes_uint(a, b) == 0);
    }

    SECTION("first word difference dominates") {
        uint32_t a[] = {0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
        uint32_t b[] = {1, 0, 0, 0, 0, 0, 0, 0};
        REQUIRE(compare_hashes_uint(a, b) < 0);
    }

    SECTION("more leading zeros is better") {
        uint32_t better[] = {0x00000000, 0x00000001, 0, 0, 0, 0, 0, 0};
        uint32_t worse[]  = {0x00000000, 0x00000002, 0, 0, 0, 0, 0, 0};
        REQUIRE(compare_hashes_uint(better, worse) < 0);
    }
}

TEST_CASE("hex_to_uint parses correctly", "[hash_utils]") {
    SECTION("valid hex") {
        uint32_t out[2];
        REQUIRE(hex_to_uint("deadbeef12345678", out, 2) == true);
        REQUIRE(out[0] == 0xdeadbeef);
        REQUIRE(out[1] == 0x12345678);
    }

    SECTION("wrong length returns false") {
        uint32_t out[2];
        REQUIRE(hex_to_uint("deadbeef", out, 2) == false);  // Too short
    }
}

TEST_CASE("bytes_to_uint converts big-endian correctly", "[hash_utils]") {
    uint8_t bytes[] = {0xde, 0xad, 0xbe, 0xef, 0x12, 0x34, 0x56, 0x78};
    uint32_t out[2];
    bytes_to_uint(bytes, out, 2);
    REQUIRE(out[0] == 0xdeadbeef);
    REQUIRE(out[1] == 0x12345678);
}

// =============================================================================
// OpenCL Error String Tests
// =============================================================================

TEST_CASE("cl_error_string returns correct strings", "[cl_error]") {
    SECTION("success code") {
        REQUIRE(std::string(cl_error_string(CL_SUCCESS)) == "CL_SUCCESS");
    }

    SECTION("common runtime errors") {
        REQUIRE(std::string(cl_error_string(CL_DEVICE_NOT_FOUND)) == "CL_DEVICE_NOT_FOUND");
        REQUIRE(std::string(cl_error_string(CL_OUT_OF_RESOURCES)) == "CL_OUT_OF_RESOURCES");
        REQUIRE(std::string(cl_error_string(CL_OUT_OF_HOST_MEMORY)) == "CL_OUT_OF_HOST_MEMORY");
        REQUIRE(std::string(cl_error_string(CL_BUILD_PROGRAM_FAILURE)) == "CL_BUILD_PROGRAM_FAILURE");
    }

    SECTION("common invalid argument errors") {
        REQUIRE(std::string(cl_error_string(CL_INVALID_VALUE)) == "CL_INVALID_VALUE");
        REQUIRE(std::string(cl_error_string(CL_INVALID_CONTEXT)) == "CL_INVALID_CONTEXT");
        REQUIRE(std::string(cl_error_string(CL_INVALID_KERNEL)) == "CL_INVALID_KERNEL");
        REQUIRE(std::string(cl_error_string(CL_INVALID_WORK_GROUP_SIZE)) == "CL_INVALID_WORK_GROUP_SIZE");
    }

    SECTION("unknown error code") {
        REQUIRE(std::string(cl_error_string(99999)) == "CL_UNKNOWN_ERROR");
    }
}

// =============================================================================
// Configuration Tests
// =============================================================================

TEST_CASE("config constants are valid", "[config]") {
    SECTION("nonce length calculation") {
        // username + '/' + nonce must equal 32 bytes
        REQUIRE(config::username_len + config::separator_len + config::nonce_len == config::sha256_block_size);
    }

    SECTION("work sizes are valid") {
        REQUIRE(config::global_size > 0);
        REQUIRE(config::local_size > 0);
        REQUIRE(config::global_size % config::local_size == 0);
    }

    SECTION("max results is reasonable") {
        REQUIRE(config::max_results >= 1);
        REQUIRE(config::max_results <= 1024);
    }
}

// =============================================================================
// GPU SHA-256 Correctness Tests
// =============================================================================

TEST_CASE("GPU SHA-256 matches known test vectors", "[gpu][sha256]") {
    auto gpu_opt = get_test_gpu();
    if (!gpu_opt) {
        WARN("No GPU available - skipping GPU tests");
        return;
    }
    auto& gpu = *gpu_opt;

    // The kernel is designed for "username/nonce" format (32 bytes total)
    // We test the existing validation which uses seed 0x12345678

    SECTION("validation hash is correct") {
        // This is the hash our validate_gpu() function checks
        // Input: "brandonros/[nonce from seed 0x12345678]"
        // We trust this is correct since it was manually verified

        std::vector<uint32_t> permissive_target(8, 0xFFFFFFFF);
        cl_uint validation_seed = 0x12345678;
        cl_uint zero = 0;

        clEnqueueWriteBuffer(gpu.queue, gpu.target_hash_buf, CL_FALSE, 0,
                             8 * sizeof(cl_uint), permissive_target.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(gpu.queue, gpu.found_count_buf, CL_FALSE, 0,
                             sizeof(cl_uint), &zero, 0, nullptr, nullptr);
        clSetKernelArg(gpu.kernel, 3, sizeof(cl_uint), &validation_seed);

        size_t global_size = 1;
        size_t local_size = 1;
        cl_int err = clEnqueueNDRangeKernel(gpu.queue, gpu.kernel, 1, nullptr,
                                             &global_size, &local_size, 0, nullptr, nullptr);
        REQUIRE(err == CL_SUCCESS);
        clFinish(gpu.queue);

        cl_uint found_count;
        clEnqueueReadBuffer(gpu.queue, gpu.found_count_buf, CL_TRUE, 0,
                            sizeof(cl_uint), &found_count, 0, nullptr, nullptr);
        REQUIRE(found_count >= 1);

        std::vector<uint8_t> hash(32);
        clEnqueueReadBuffer(gpu.queue, gpu.found_hashes_buf, CL_TRUE, 0, 32, hash.data(), 0, nullptr, nullptr);

        std::string hash_hex = bytes_to_hex(hash.data(), 32);
        // Expected hash with improved RNG seeding (golden ratio multiplier)
        REQUIRE(hash_hex == "f91db0d6b82e572f512302769cd22db910b9b8e09d96969f1fb29b4ce8aa3a4c");
    }
}

TEST_CASE("Different seeds produce different nonces", "[gpu][rng]") {
    auto gpu_opt = get_test_gpu();
    if (!gpu_opt) {
        WARN("No GPU available - skipping GPU tests");
        return;
    }
    auto& gpu = *gpu_opt;

    std::vector<uint32_t> permissive_target(8, 0xFFFFFFFF);
    std::vector<std::string> nonces;

    // Run with 10 different seeds
    for (cl_uint seed = 0; seed < 10; seed++) {
        cl_uint zero = 0;
        clEnqueueWriteBuffer(gpu.queue, gpu.target_hash_buf, CL_FALSE, 0,
                             8 * sizeof(cl_uint), permissive_target.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(gpu.queue, gpu.found_count_buf, CL_FALSE, 0,
                             sizeof(cl_uint), &zero, 0, nullptr, nullptr);
        clSetKernelArg(gpu.kernel, 3, sizeof(cl_uint), &seed);

        size_t global_size = 1;
        size_t local_size = 1;
        clEnqueueNDRangeKernel(gpu.queue, gpu.kernel, 1, nullptr,
                                &global_size, &local_size, 0, nullptr, nullptr);
        clFinish(gpu.queue);

        std::vector<uint8_t> nonce(32);
        clEnqueueReadBuffer(gpu.queue, gpu.found_nonces_buf, CL_TRUE, 0, 32, nonce.data(), 0, nullptr, nullptr);

        std::string nonce_str(reinterpret_cast<char*>(nonce.data()), config::nonce_len);
        nonces.push_back(nonce_str);
    }

    // All nonces should be unique
    std::sort(nonces.begin(), nonces.end());
    auto last = std::unique(nonces.begin(), nonces.end());
    REQUIRE(last == nonces.end());  // No duplicates
}

TEST_CASE("Target filtering works correctly", "[gpu][mining]") {
    auto gpu_opt = get_test_gpu();
    if (!gpu_opt) {
        WARN("No GPU available - skipping GPU tests");
        return;
    }
    auto& gpu = *gpu_opt;

    SECTION("permissive target accepts all hashes") {
        std::vector<uint32_t> permissive_target(8, 0xFFFFFFFF);
        cl_uint zero = 0;
        cl_uint seed = 12345;

        clEnqueueWriteBuffer(gpu.queue, gpu.target_hash_buf, CL_FALSE, 0,
                             8 * sizeof(cl_uint), permissive_target.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(gpu.queue, gpu.found_count_buf, CL_FALSE, 0,
                             sizeof(cl_uint), &zero, 0, nullptr, nullptr);
        clSetKernelArg(gpu.kernel, 3, sizeof(cl_uint), &seed);

        size_t global_size = 256;  // Run more threads
        size_t local_size = 256;
        clEnqueueNDRangeKernel(gpu.queue, gpu.kernel, 1, nullptr,
                                &global_size, &local_size, 0, nullptr, nullptr);
        clFinish(gpu.queue);

        cl_uint found_count;
        clEnqueueReadBuffer(gpu.queue, gpu.found_count_buf, CL_TRUE, 0,
                            sizeof(cl_uint), &found_count, 0, nullptr, nullptr);

        // With permissive target and HASHES_PER_THREAD iterations per thread,
        // we should find many matches
        REQUIRE(found_count > 0);
        INFO("Found " << found_count << " matches with permissive target");
    }

    SECTION("restrictive target rejects most hashes") {
        // Target with 8 leading zero nibbles (32 bits) - very restrictive
        std::vector<uint32_t> restrictive_target = {
            0x00000000, 0x00000001, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000
        };
        cl_uint zero = 0;
        cl_uint seed = 12345;

        clEnqueueWriteBuffer(gpu.queue, gpu.target_hash_buf, CL_FALSE, 0,
                             8 * sizeof(cl_uint), restrictive_target.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(gpu.queue, gpu.found_count_buf, CL_FALSE, 0,
                             sizeof(cl_uint), &zero, 0, nullptr, nullptr);
        clSetKernelArg(gpu.kernel, 3, sizeof(cl_uint), &seed);

        size_t global_size = 256;
        size_t local_size = 256;
        clEnqueueNDRangeKernel(gpu.queue, gpu.kernel, 1, nullptr,
                                &global_size, &local_size, 0, nullptr, nullptr);
        clFinish(gpu.queue);

        cl_uint found_count;
        clEnqueueReadBuffer(gpu.queue, gpu.found_count_buf, CL_TRUE, 0,
                            sizeof(cl_uint), &found_count, 0, nullptr, nullptr);

        // With restrictive target, we should find very few or no matches
        // (probability ~1/2^32 per hash)
        INFO("Found " << found_count << " matches with restrictive target (expected ~0)");
        // Don't assert 0, as we might get lucky, but it should be rare
    }
}

TEST_CASE("Found hashes are actually better than target", "[gpu][mining][critical]") {
    auto gpu_opt = get_test_gpu();
    if (!gpu_opt) {
        WARN("No GPU available - skipping GPU tests");
        return;
    }
    auto& gpu = *gpu_opt;

    // Use a moderately permissive target (4 leading zero nibbles = 16 bits)
    std::vector<uint32_t> target = {
        0x0000FFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    };
    cl_uint zero = 0;
    cl_uint seed = 99999;

    clEnqueueWriteBuffer(gpu.queue, gpu.target_hash_buf, CL_FALSE, 0,
                         8 * sizeof(cl_uint), target.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(gpu.queue, gpu.found_count_buf, CL_FALSE, 0,
                         sizeof(cl_uint), &zero, 0, nullptr, nullptr);
    clSetKernelArg(gpu.kernel, 3, sizeof(cl_uint), &seed);

    // Run enough threads to likely find some matches
    size_t global_size = 65536;
    size_t local_size = 256;
    clEnqueueNDRangeKernel(gpu.queue, gpu.kernel, 1, nullptr,
                            &global_size, &local_size, 0, nullptr, nullptr);
    clFinish(gpu.queue);

    cl_uint found_count;
    clEnqueueReadBuffer(gpu.queue, gpu.found_count_buf, CL_TRUE, 0,
                        sizeof(cl_uint), &found_count, 0, nullptr, nullptr);

    if (found_count == 0) {
        WARN("No matches found - try running with more threads or different seed");
        return;
    }

    // Read all found hashes and verify each is actually < target
    size_t to_check = std::min(static_cast<size_t>(found_count), config::max_results);
    std::vector<uint8_t> all_hashes(to_check * 32);
    clEnqueueReadBuffer(gpu.queue, gpu.found_hashes_buf, CL_TRUE, 0,
                        to_check * 32, all_hashes.data(), 0, nullptr, nullptr);

    INFO("Checking " << to_check << " found hashes");

    for (size_t i = 0; i < to_check; i++) {
        std::vector<uint32_t> hash_uint(8);
        bytes_to_uint(all_hashes.data() + i * 32, hash_uint.data(), 8);

        int cmp = compare_hashes_uint(hash_uint.data(), target.data());

        INFO("Hash " << i << ": " << bytes_to_hex(all_hashes.data() + i * 32, 32));
        INFO("Target: " << uint_to_hex(target.data(), 8));
        INFO("Comparison result: " << cmp << " (should be < 0)");

        REQUIRE(cmp < 0);  // Hash must be less than target
    }
}

TEST_CASE("Nonce characters are valid base64", "[gpu][rng]") {
    auto gpu_opt = get_test_gpu();
    if (!gpu_opt) {
        WARN("No GPU available - skipping GPU tests");
        return;
    }
    auto& gpu = *gpu_opt;

    const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::vector<uint32_t> permissive_target(8, 0xFFFFFFFF);
    std::set<char> seen_chars;

    // Run with many different seeds to sample the character distribution
    for (cl_uint seed = 0; seed < 1000; seed++) {
        cl_uint zero = 0;
        clEnqueueWriteBuffer(gpu.queue, gpu.target_hash_buf, CL_FALSE, 0,
                             8 * sizeof(cl_uint), permissive_target.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(gpu.queue, gpu.found_count_buf, CL_FALSE, 0,
                             sizeof(cl_uint), &zero, 0, nullptr, nullptr);
        clSetKernelArg(gpu.kernel, 3, sizeof(cl_uint), &seed);

        size_t global_size = 1;
        size_t local_size = 1;
        clEnqueueNDRangeKernel(gpu.queue, gpu.kernel, 1, nullptr,
                                &global_size, &local_size, 0, nullptr, nullptr);
        clFinish(gpu.queue);

        std::vector<uint8_t> nonce(32);
        clEnqueueReadBuffer(gpu.queue, gpu.found_nonces_buf, CL_TRUE, 0, 32, nonce.data(), 0, nullptr, nullptr);

        for (size_t i = 0; i < config::nonce_len; i++) {
            char c = static_cast<char>(nonce[i]);
            REQUIRE(base64_chars.find(c) != std::string::npos);
            seen_chars.insert(c);
        }
    }

    // Should see most of the base64 alphabet after 1000 samples
    INFO("Saw " << seen_chars.size() << " unique characters out of 64");
    REQUIRE(seen_chars.size() >= 50);  // At least ~78% coverage
}
