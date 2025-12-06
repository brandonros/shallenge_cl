// SHA-256 implementation for OpenCL
// Optimized for 32-byte input (single block)

// SHA-256 round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
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

// SHA-256 initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
__constant uint H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// Rotate right
inline uint rotr32(uint x, uint n) {
    return (x >> n) | (x << (32 - n));
}

// SHA-256 helper functions
inline uint ch(uint x, uint y, uint z) {
    return (x & y) ^ (~x & z);
}

inline uint maj(uint x, uint y, uint z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

inline uint big_sigma0(uint x) {
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}

inline uint big_sigma1(uint x) {
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}

inline uint small_sigma0(uint x) {
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}

inline uint small_sigma1(uint x) {
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

// SHA-256 for exactly 32-byte input
// Input: 32 bytes in 'input' array
// Output: 32 bytes in 'output' array
void sha256_32(const uchar* input, uchar* output) {
    uint w[64];

    // Copy 32 bytes of input to first 8 words (big-endian)
    for (int i = 0; i < 8; i++) {
        w[i] = ((uint)input[i*4] << 24) |
               ((uint)input[i*4+1] << 16) |
               ((uint)input[i*4+2] << 8) |
               ((uint)input[i*4+3]);
    }

    // Padding for 32-byte (256-bit) message:
    // Byte 32: 0x80 (1 bit followed by zeros)
    // Bytes 33-61: 0x00
    // Bytes 62-63: length in bits = 256 = 0x0100
    w[8] = 0x80000000;
    w[9] = 0;
    w[10] = 0;
    w[11] = 0;
    w[12] = 0;
    w[13] = 0;
    w[14] = 0;
    w[15] = 256;  // 32 bytes = 256 bits

    // Extend the first 16 words into remaining 48 words
    for (int i = 16; i < 64; i++) {
        w[i] = small_sigma1(w[i-2]) + w[i-7] + small_sigma0(w[i-15]) + w[i-16];
    }

    // Initialize working variables
    uint a = H0[0];
    uint b = H0[1];
    uint c = H0[2];
    uint d = H0[3];
    uint e = H0[4];
    uint f = H0[5];
    uint g = H0[6];
    uint h = H0[7];

    // Main compression loop - 64 rounds
    for (int i = 0; i < 64; i++) {
        uint t1 = h + big_sigma1(e) + ch(e, f, g) + K[i] + w[i];
        uint t2 = big_sigma0(a) + maj(a, b, c);

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Add to initial hash values
    uint hash[8];
    hash[0] = H0[0] + a;
    hash[1] = H0[1] + b;
    hash[2] = H0[2] + c;
    hash[3] = H0[3] + d;
    hash[4] = H0[4] + e;
    hash[5] = H0[5] + f;
    hash[6] = H0[6] + g;
    hash[7] = H0[7] + h;

    // Convert to bytes (big-endian)
    for (int i = 0; i < 8; i++) {
        output[i*4]   = (hash[i] >> 24) & 0xFF;
        output[i*4+1] = (hash[i] >> 16) & 0xFF;
        output[i*4+2] = (hash[i] >> 8) & 0xFF;
        output[i*4+3] = hash[i] & 0xFF;
    }
}

// Compare two 32-byte hashes lexicographically
// Returns: -1 if a < b, 0 if a == b, 1 if a > b
int compare_hashes(const uchar* a, const uchar* b) {
    for (int i = 0; i < 32; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}
