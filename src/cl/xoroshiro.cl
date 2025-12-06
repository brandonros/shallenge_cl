// Xoroshiro128** RNG implementation for OpenCL
// Used for generating random base64 nonces

__constant uchar BASE64_CHARS[64] = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'
};

// Rotate left for 64-bit
inline ulong rotl64(ulong x, int k) {
    return (x << k) | (x >> (64 - k));
}

// Splitmix64 - used to initialize xoroshiro state from a single seed
ulong splitmix64(ulong x) {
    x += 0x9e3779b97f4a7c15UL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9UL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebUL;
    return x ^ (x >> 31);
}

// Xoroshiro128** state
typedef struct {
    ulong s0;
    ulong s1;
} xoroshiro128ss_state;

// Initialize xoroshiro128** from a seed
void xoroshiro128ss_init(xoroshiro128ss_state* state, ulong seed) {
    state->s0 = splitmix64(seed);
    state->s1 = splitmix64(state->s0);
}

// Generate next 64-bit random number
ulong xoroshiro128ss_next(xoroshiro128ss_state* state) {
    ulong s0 = state->s0;
    ulong s1 = state->s1;

    // Result calculation: rotl(s0 * 5, 7) * 9
    ulong result = rotl64(s0 * 5, 7) * 9;

    // State update
    s1 ^= s0;
    state->s0 = rotl64(s0, 24) ^ s1 ^ (s1 << 16);
    state->s1 = rotl64(s1, 37);

    return result;
}

// Generate next 32-bit random number
uint xoroshiro128ss_next_u32(xoroshiro128ss_state* state) {
    return (uint)(xoroshiro128ss_next(state) >> 32);
}

// Generate a base64 nonce of given length
// thread_idx and rng_seed are combined to create unique per-thread randomness
void generate_base64_nonce(size_t thread_idx, ulong rng_seed, uchar* nonce, size_t nonce_len) {
    // Mix seed with thread index
    ulong mixed_seed = splitmix64(rng_seed + (ulong)thread_idx);

    // Initialize RNG
    xoroshiro128ss_state rng;
    xoroshiro128ss_init(&rng, mixed_seed);

    // Generate nonce characters
    for (size_t i = 0; i < nonce_len; i++) {
        uint idx = xoroshiro128ss_next_u32(&rng) % 64;
        nonce[i] = BASE64_CHARS[idx];
    }
}
