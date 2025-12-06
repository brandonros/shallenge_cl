#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <string>
#include <sstream>
#include <iomanip>

namespace shallenge {

// Convert bytes to hex string
[[nodiscard]] inline std::string bytes_to_hex(const uint8_t* data, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; i++) {
        ss << std::setw(2) << static_cast<int>(data[i]);
    }
    return ss.str();
}

// Count leading zero nibbles in hex string
[[nodiscard]] inline int count_leading_zeros(const std::string& hex_string) {
    int count = 0;
    for (char c : hex_string) {
        if (c == '0') count++;
        else break;
    }
    return count;
}

// Convert uint32_t array to hex string (big-endian)
[[nodiscard]] inline std::string uint_to_hex(const uint32_t* data, size_t count) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < count; i++) {
        ss << std::setw(8) << data[i];
    }
    return ss.str();
}

// Parse hex string to uint32_t array (big-endian)
[[nodiscard]] inline bool hex_to_uint(const std::string& hex, uint32_t* out, size_t out_count) {
    if (hex.length() != out_count * 8) return false;
    for (size_t i = 0; i < out_count; i++) {
        unsigned int word;
        if (sscanf(hex.c_str() + i * 8, "%8x", &word) != 1) return false;
        out[i] = static_cast<uint32_t>(word);
    }
    return true;
}

// Convert bytes to uint32_t array (big-endian)
inline void bytes_to_uint(const uint8_t* bytes, uint32_t* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = (static_cast<uint32_t>(bytes[i*4]) << 24) |
                 (static_cast<uint32_t>(bytes[i*4+1]) << 16) |
                 (static_cast<uint32_t>(bytes[i*4+2]) << 8) |
                 static_cast<uint32_t>(bytes[i*4+3]);
    }
}

// Compare two 8-word hashes lexicographically
// Returns: -1 if a < b, 0 if a == b, 1 if a > b
[[nodiscard]] inline int compare_hashes_uint(const uint32_t* a, const uint32_t* b) {
    for (int i = 0; i < 8; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

} // namespace shallenge
