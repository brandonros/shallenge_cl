// SHA-256 implementation for OpenCL
// Optimized for 32-byte input (single block)
// Performance optimizations:
// - OpenCL built-ins (rotate, bitselect)
// - Macros for zero function-call overhead
// - 16-word ring buffer to reduce register pressure
// - Loop unrolling hints

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

// SHA-256 for exactly 32-byte input
// Uses 16-word ring buffer to reduce register pressure
void sha256_32(const uchar* restrict input, uchar* restrict output) {
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

    // Main compression loop - 64 rounds with ring buffer
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint wi;
        int j = i & 0xF;

        if (i < 16) {
            wi = w[j];
        } else {
            wi = w[j] = SSIG1(w[(j + 14) & 0xF]) +
                        w[(j + 9) & 0xF] +
                        SSIG0(w[(j + 1) & 0xF]) +
                        w[j];
        }

        uint t1 = h + BSIG1(e) + CH(e, f, g) + K[i] + wi;
        uint t2 = BSIG0(a) + MAJ(a, b, c);

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Add initial hash values and store directly (fused)
    uint h0 = 0x6a09e667u + a;
    uint h1 = 0xbb67ae85u + b;
    uint h2 = 0x3c6ef372u + c;
    uint h3 = 0xa54ff53au + d;
    uint h4 = 0x510e527fu + e;
    uint h5 = 0x9b05688cu + f;
    uint h6 = 0x1f83d9abu + g;
    uint h7 = 0x5be0cd19u + h;

    // Store output (big-endian)
    output[0]  = (h0 >> 24); output[1]  = (h0 >> 16); output[2]  = (h0 >> 8); output[3]  = h0;
    output[4]  = (h1 >> 24); output[5]  = (h1 >> 16); output[6]  = (h1 >> 8); output[7]  = h1;
    output[8]  = (h2 >> 24); output[9]  = (h2 >> 16); output[10] = (h2 >> 8); output[11] = h2;
    output[12] = (h3 >> 24); output[13] = (h3 >> 16); output[14] = (h3 >> 8); output[15] = h3;
    output[16] = (h4 >> 24); output[17] = (h4 >> 16); output[18] = (h4 >> 8); output[19] = h4;
    output[20] = (h5 >> 24); output[21] = (h5 >> 16); output[22] = (h5 >> 8); output[23] = h5;
    output[24] = (h6 >> 24); output[25] = (h6 >> 16); output[26] = (h6 >> 8); output[27] = h6;
    output[28] = (h7 >> 24); output[29] = (h7 >> 16); output[30] = (h7 >> 8); output[31] = h7;
}

// Compare two 32-byte hashes lexicographically
// Returns: -1 if a < b, 0 if a == b, 1 if a > b
int compare_hashes(const uchar* restrict a, const uchar* restrict b) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}
