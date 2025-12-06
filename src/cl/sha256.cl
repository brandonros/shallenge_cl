// SHA-256 implementation for OpenCL
// Optimized for 32-byte input (single block)
// Performance optimizations:
// - OpenCL built-ins (rotate, bitselect)
// - Macros for zero function-call overhead
// - 16-word ring buffer to reduce register pressure
// - Split loops to eliminate branches
// - Word-based comparison (8 uint vs 32 uchar)
// - Early exit comparison

// SHA-256 round constants
__constant uint K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 helper macros using OpenCL built-ins
// ch(x,y,z) = (x & y) ^ (~x & z) = bitselect(z, y, x)
#define CH(x, y, z)    bitselect((z), (y), (x))
#define MAJ(x, y, z)   (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

// Big sigma functions (rotate uses left rotation, so we compute 32-n)
#define BSIG0(x)       (rotate((x), 30u) ^ rotate((x), 19u) ^ rotate((x), 10u))
#define BSIG1(x)       (rotate((x), 26u) ^ rotate((x), 21u) ^ rotate((x), 7u))

// Small sigma functions
#define SSIG0(x)       (rotate((x), 25u) ^ rotate((x), 14u) ^ ((x) >> 3))
#define SSIG1(x)       (rotate((x), 15u) ^ rotate((x), 13u) ^ ((x) >> 10))

// SHA-256 round macro to avoid code duplication
#define SHA256_ROUND(a, b, c, d, e, f, g, h, ki, wi) \
    do { \
        uint t1 = (h) + BSIG1(e) + CH(e, f, g) + (ki) + (wi); \
        uint t2 = BSIG0(a) + MAJ(a, b, c); \
        (h) = (g); \
        (g) = (f); \
        (f) = (e); \
        (e) = (d) + t1; \
        (d) = (c); \
        (c) = (b); \
        (b) = (a); \
        (a) = t1 + t2; \
    } while(0)

// SHA-256 for exactly 32-byte input, returns hash as uint[8] (big-endian words)
// This avoids the costly byte-by-byte output conversion
inline void sha256_32_uint(const uchar* restrict input, uint* restrict output) {
    uint w[16];

    // Load first 8 words from input (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        w[i] = ((uint)input[i*4] << 24) |
               ((uint)input[i*4+1] << 16) |
               ((uint)input[i*4+2] << 8) |
               ((uint)input[i*4+3]);
    }

    // Padding for 32-byte message
    w[8]  = 0x80000000u;
    w[9]  = 0u;
    w[10] = 0u;
    w[11] = 0u;
    w[12] = 0u;
    w[13] = 0u;
    w[14] = 0u;
    w[15] = 256u;  // 32 bytes = 256 bits

    // Initialize working variables from initial hash values
    uint a = 0x6a09e667u;
    uint b = 0xbb67ae85u;
    uint c = 0x3c6ef372u;
    uint d = 0xa54ff53au;
    uint e = 0x510e527fu;
    uint f = 0x9b05688cu;
    uint g = 0x1f83d9abu;
    uint h = 0x5be0cd19u;

    // Rounds 0-15: Use message words directly (no message schedule computation)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        SHA256_ROUND(a, b, c, d, e, f, g, h, K[i], w[i]);
    }

    // Rounds 16-63: Compute message schedule on the fly with ring buffer
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        int j = i & 0xF;
        w[j] = SSIG1(w[(j + 14) & 0xF]) +
               w[(j + 9) & 0xF] +
               SSIG0(w[(j + 1) & 0xF]) +
               w[j];
        SHA256_ROUND(a, b, c, d, e, f, g, h, K[i], w[j]);
    }

    // Add initial hash values and store as uint[8]
    output[0] = 0x6a09e667u + a;
    output[1] = 0xbb67ae85u + b;
    output[2] = 0x3c6ef372u + c;
    output[3] = 0xa54ff53au + d;
    output[4] = 0x510e527fu + e;
    output[5] = 0x9b05688cu + f;
    output[6] = 0x1f83d9abu + g;
    output[7] = 0x5be0cd19u + h;
}

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
