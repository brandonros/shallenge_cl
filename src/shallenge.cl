// Shallenge OpenCL Kernel - Consolidated
// SHA-256 mining for https://shallenge.quirino.net/

// ============================================================================
// SHA-256 Implementation (optimized for 32-byte input)
// ============================================================================

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

#define CH(x, y, z)    bitselect((z), (y), (x))
#define MAJ(x, y, z)   (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define BSIG0(x)       (rotate((x), 30u) ^ rotate((x), 19u) ^ rotate((x), 10u))
#define BSIG1(x)       (rotate((x), 26u) ^ rotate((x), 21u) ^ rotate((x), 7u))
#define SSIG0(x)       (rotate((x), 25u) ^ rotate((x), 14u) ^ ((x) >> 3))
#define SSIG1(x)       (rotate((x), 15u) ^ rotate((x), 13u) ^ ((x) >> 10))

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

inline void sha256_32_uint(const uchar* restrict input, uint* restrict output) {
    uint w[16];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        w[i] = ((uint)input[i*4] << 24) |
               ((uint)input[i*4+1] << 16) |
               ((uint)input[i*4+2] << 8) |
               ((uint)input[i*4+3]);
    }

    w[8]  = 0x80000000u;
    w[9]  = 0u;
    w[10] = 0u;
    w[11] = 0u;
    w[12] = 0u;
    w[13] = 0u;
    w[14] = 0u;
    w[15] = 256u;

    uint a = 0x6a09e667u;
    uint b = 0xbb67ae85u;
    uint c = 0x3c6ef372u;
    uint d = 0xa54ff53au;
    uint e = 0x510e527fu;
    uint f = 0x9b05688cu;
    uint g = 0x1f83d9abu;
    uint h = 0x5be0cd19u;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        SHA256_ROUND(a, b, c, d, e, f, g, h, K[i], w[i]);
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        int j = i & 0xF;
        w[j] = SSIG1(w[(j + 14) & 0xF]) +
               w[(j + 9) & 0xF] +
               SSIG0(w[(j + 1) & 0xF]) +
               w[j];
        SHA256_ROUND(a, b, c, d, e, f, g, h, K[i], w[j]);
    }

    output[0] = 0x6a09e667u + a;
    output[1] = 0xbb67ae85u + b;
    output[2] = 0x3c6ef372u + c;
    output[3] = 0xa54ff53au + d;
    output[4] = 0x510e527fu + e;
    output[5] = 0x9b05688cu + f;
    output[6] = 0x1f83d9abu + g;
    output[7] = 0x5be0cd19u + h;
}

// ============================================================================
// Hash Comparison Utilities
// ============================================================================

inline int compare_hashes_uint(const uint* restrict a, const uint* restrict b) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

inline int is_hash_better(const uint* restrict hash, __local const uint* restrict target) {
    if (hash[0] > target[0]) return 0;
    if (hash[0] < target[0]) return 1;

    #pragma unroll
    for (int i = 1; i < 8; i++) {
        if (hash[i] > target[i]) return 0;
        if (hash[i] < target[i]) return 1;
    }
    return 0;
}

// ============================================================================
// RNG and Nonce Generation (xoroshiro64**)
// ============================================================================

__constant uchar BASE64_CHARS[64] = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'
};

#define ROTL32(x, k) (((x) << (k)) | ((x) >> (32 - (k))))

inline uint splitmix32(uint x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x85ebca6bu;
    x = (x ^ (x >> 13)) * 0xc2b2ae35u;
    return x ^ (x >> 16);
}

inline void init_rng_state(uint thread_idx, uint rng_seed_lo, uint rng_seed_hi, uint* s0, uint* s1) {
    *s0 = splitmix32(rng_seed_lo ^ thread_idx);
    *s1 = splitmix32(rng_seed_hi ^ (thread_idx * 0x9e3779b9u));
    if (*s0 == 0 && *s1 == 0) *s0 = 1;
}

inline uint xoroshiro64_next(uint* s0, uint* s1) {
    uint result = ROTL32(*s0 * 0x9E3779BBu, 5) * 5;
    uint t = *s1 ^ *s0;
    *s0 = ROTL32(*s0, 26) ^ t ^ (t << 9);
    *s1 = ROTL32(t, 13);
    return result;
}

inline void generate_nonce_from_state(uint* s0, uint* s1, uchar* restrict nonce, size_t nonce_len) {
    size_t i = 0;
    while (i < nonce_len) {
        uint bits = xoroshiro64_next(s0, s1);
        #pragma unroll
        for (int j = 0; j < 5 && i < nonce_len; j++, i++) {
            nonce[i] = BASE64_CHARS[bits & 63];
            bits >>= 6;
        }
    }
}

// ============================================================================
// Main Mining Kernel
// ============================================================================

#define MAX_RESULTS 64

__kernel void shallenge_mine(
    __global const uchar* restrict username,
    uint username_len,
    __global const uint* restrict target_hash,
    uint rng_seed_lo,
    uint rng_seed_hi,
    __global uint* restrict found_count,
    __global uchar* restrict found_hashes,
    __global uchar* restrict found_nonces,
    __global uint* restrict found_thread_ids,
    __local uint* restrict target_local
) {
    uint thread_idx = (uint)get_global_id(0);
    uint lid = get_local_id(0);
    uint nonce_len = 31 - username_len;

    if (lid < 8) {
        target_local[lid] = target_hash[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint s0, s1;
    init_rng_state(thread_idx, rng_seed_lo, rng_seed_hi, &s0, &s1);

    uchar input[32];
    for (uint i = 0; i < username_len; i++) {
        input[i] = username[i];
    }
    input[username_len] = '/';

    for (int iter = 0; iter < HASHES_PER_THREAD; iter++) {
        generate_nonce_from_state(&s0, &s1, &input[username_len + 1], nonce_len);

        uint hash[8];
        sha256_32_uint(input, hash);

        if (is_hash_better(hash, target_local)) {
            uint slot = atomic_inc(found_count);

            if (slot < MAX_RESULTS) {
                __global uchar* hash_out = found_hashes + slot * 32;
                __global uchar* nonce_out = found_nonces + slot * 32;

                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    hash_out[i*4]     = (hash[i] >> 24);
                    hash_out[i*4 + 1] = (hash[i] >> 16);
                    hash_out[i*4 + 2] = (hash[i] >> 8);
                    hash_out[i*4 + 3] = hash[i];
                }

                for (uint i = 0; i < nonce_len; i++) {
                    nonce_out[i] = input[username_len + 1 + i];
                }

                found_thread_ids[slot] = thread_idx;
            }
        }
    }
}
