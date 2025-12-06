#pragma once

#include <cstddef>
#include <cstdint>

namespace shallenge {
namespace config {

// Compile-time configuration (from Makefile -D flags)
// These macros must be defined at compile time
constexpr const char* username = DEFAULT_USERNAME;
constexpr size_t global_size = GLOBAL_SIZE;
constexpr size_t local_size = LOCAL_SIZE;
constexpr size_t hashes_per_thread = HASHES_PER_THREAD;

// Derived constants
constexpr size_t username_len = sizeof(DEFAULT_USERNAME) - 1;
constexpr size_t sha256_block_size = 32;
constexpr size_t separator_len = 1;
constexpr size_t nonce_len = sha256_block_size - (username_len + separator_len);
constexpr size_t max_results = 64;

// Initial target hash (8 leading zero nibbles)
constexpr uint32_t initial_target[8] = {
    0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

} // namespace config
} // namespace shallenge
