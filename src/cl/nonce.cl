// xoroshiro64** RNG - 32-bit version for GPU efficiency
// Batched extraction: 5 base64 chars per 32-bit value (5 * 6 = 30 bits)
// For 21-char nonce: 5 RNG calls instead of 21

__constant uchar BASE64_CHARS[64] = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'
};

// Rotate left for 32-bit
#define ROTL32(x, k) (((x) << (k)) | ((x) >> (32 - (k))))

// Splitmix32 - initialize state from seed
inline uint splitmix32(uint x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x85ebca6bu;
    x = (x ^ (x >> 13)) * 0xc2b2ae35u;
    return x ^ (x >> 16);
}

// Initialize RNG state once per thread (fully 32-bit)
inline void init_rng_state(uint thread_idx, uint rng_seed, uint* s0, uint* s1) {
    uint seed = splitmix32(rng_seed ^ thread_idx);
    *s0 = seed;
    seed = splitmix32(seed);
    *s1 = seed;
    // Ensure non-zero state
    if (*s0 == 0 && *s1 == 0) *s0 = 1;
}

// xoroshiro64** next value
inline uint xoroshiro64_next(uint* s0, uint* s1) {
    uint result = ROTL32(*s0 * 0x9E3779BBu, 5) * 5;

    uint t = *s1 ^ *s0;
    *s0 = ROTL32(*s0, 26) ^ t ^ (t << 9);
    *s1 = ROTL32(t, 13);

    return result;
}

// Generate nonce with batched extraction
// Each 32-bit value provides 5 base64 characters (30 bits used, 2 discarded)
inline void generate_nonce_from_state(uint* s0, uint* s1, uchar* restrict nonce, size_t nonce_len) {
    size_t i = 0;
    while (i < nonce_len) {
        uint bits = xoroshiro64_next(s0, s1);

        // Extract up to 5 characters (6 bits each) from 32 bits
        #pragma unroll
        for (int j = 0; j < 5 && i < nonce_len; j++, i++) {
            nonce[i] = BASE64_CHARS[bits & 63];
            bits >>= 6;
        }
    }
}
