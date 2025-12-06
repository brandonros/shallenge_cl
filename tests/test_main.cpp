#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "core/hash_utils.hpp"
#include "config.hpp"

using namespace shallenge;

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
