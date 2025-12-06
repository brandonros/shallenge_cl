// Xoroshiro128** RNG implementation for OpenCL
// Used for generating random base64 nonces
// Optimized with macros and inlining

__constant uchar BASE64_CHARS[64] = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'
};

// Rotate left for 64-bit using macro
#define ROTL64(x, k) (((x) << (k)) | ((x) >> (64 - (k))))

// Splitmix64 - used to initialize xoroshiro state from a single seed
inline ulong splitmix64(ulong x) {
    x += 0x9e3779b97f4a7c15UL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9UL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebUL;
    return x ^ (x >> 31);
}

// Generate a base64 nonce of fixed length (21 chars)
// Inlined xoroshiro128** for maximum performance
inline void generate_base64_nonce(size_t thread_idx, ulong rng_seed, uchar* restrict nonce, size_t nonce_len) {
    // Mix seed with thread index using splitmix64
    ulong mixed = rng_seed + (ulong)thread_idx;
    mixed += 0x9e3779b97f4a7c15UL;
    mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9UL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebUL;
    mixed ^= (mixed >> 31);

    // Initialize xoroshiro128** state inline
    ulong s0 = mixed;
    s0 += 0x9e3779b97f4a7c15UL;
    s0 = (s0 ^ (s0 >> 30)) * 0xbf58476d1ce4e5b9UL;
    s0 = (s0 ^ (s0 >> 27)) * 0x94d049bb133111ebUL;
    s0 ^= (s0 >> 31);

    ulong s1 = s0;
    s1 += 0x9e3779b97f4a7c15UL;
    s1 = (s1 ^ (s1 >> 30)) * 0xbf58476d1ce4e5b9UL;
    s1 = (s1 ^ (s1 >> 27)) * 0x94d049bb133111ebUL;
    s1 ^= (s1 >> 31);

    // Generate nonce characters
    #pragma unroll
    for (size_t i = 0; i < nonce_len; i++) {
        // xoroshiro128** next: rotl(s0 * 5, 7) * 9
        ulong result = ROTL64(s0 * 5, 7) * 9;

        // State update
        ulong t = s1 ^ s0;
        s0 = ROTL64(s0, 24) ^ t ^ (t << 16);
        s1 = ROTL64(t, 37);

        // Extract index (use upper 32 bits, then mod 64)
        uint idx = ((uint)(result >> 32)) & 63;  // Faster than % 64 since 64 is power of 2
        nonce[i] = BASE64_CHARS[idx];
    }
}
