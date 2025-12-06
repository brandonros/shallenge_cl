// Hash utility functions for 256-bit values
// Generic operations on uint[8] arrays (big-endian representation)

// Compare two hashes stored as uint[8] (big-endian words)
// Returns: -1 if a < b, 0 if a == b, 1 if a > b
// Only 8 comparisons instead of 32
inline int compare_hashes_uint(const uint* restrict a, const uint* restrict b) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

// Fast check: is hash potentially better than target?
// Returns 1 if hash < target (better), 0 if hash >= target (worse or equal)
// Uses early exit - most hashes fail on first word
inline int is_hash_better(const uint* restrict hash, const uint* restrict target) {
    // Check first word - this eliminates ~99.9999% of hashes
    if (hash[0] > target[0]) return 0;
    if (hash[0] < target[0]) return 1;

    // First word tied, check remaining words
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        if (hash[i] > target[i]) return 0;
        if (hash[i] < target[i]) return 1;
    }
    return 0;  // Equal means not better
}

// Convert uint[8] hash to uchar[32] for output (only called for winners)
inline void hash_uint_to_bytes(const uint* restrict hash, uchar* restrict output) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        output[i*4]     = (hash[i] >> 24);
        output[i*4 + 1] = (hash[i] >> 16);
        output[i*4 + 2] = (hash[i] >> 8);
        output[i*4 + 3] = hash[i];
    }
}
