// Shallenge mining kernel
// Note: This file is concatenated after sha256.cl, util.cl, and nonce.cl

// HASHES_PER_THREAD is passed via -D at kernel compile time
// No fallback - kernel build will fail if not defined

__kernel void shallenge_mine(
    __global const uchar* restrict username,
    uint username_len,
    __global const uint* restrict target_hash,    // 8 uints (32 bytes as big-endian words)
    uint rng_seed,                                 // 32-bit random seed from host
    __global uint* restrict found_count,           // atomic counter for matches found
    __global uchar* restrict found_hash,           // 32 bytes - best hash found
    __global uchar* restrict found_nonce,          // winning nonce (32 - username_len - 1 bytes)
    __global uint* restrict found_thread_idx,      // which thread found it
    __local uint* restrict target_local            // local memory for target hash (8 uints)
) {
    uint thread_idx = (uint)get_global_id(0);      // 32-bit thread index
    int lid = get_local_id(0);

    // Calculate nonce length from username length (total input must be 32 bytes)
    uint nonce_len = 31 - username_len;  // 32 - username_len - 1 (for '/')

    // Load target hash into local memory (first 8 threads of each work-group)
    if (lid < 8) {
        target_local[lid] = target_hash[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Initialize RNG state ONCE per thread (fully 32-bit)
    uint s0, s1;
    init_rng_state(thread_idx, rng_seed, &s0, &s1);

    // Prepare input buffer with username prefix (doesn't change)
    uchar input[32];
    for (uint i = 0; i < username_len; i++) {
        input[i] = username[i];
    }
    input[username_len] = '/';

    // Inner loop - hash multiple times per thread
    for (int iter = 0; iter < HASHES_PER_THREAD; iter++) {
        // Generate nonce directly into input buffer (no separate nonce array)
        generate_nonce_from_state(&s0, &s1, &input[username_len + 1], nonce_len);

        // Compute SHA-256
        uint hash[8];
        sha256_32_uint(input, hash);

        // Check if this hash is better (compare directly against local memory)
        if (is_hash_better(hash, target_local)) {
            atomic_inc(found_count);

            // Write hash bytes directly to global memory (no temp array)
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                found_hash[i*4]     = (hash[i] >> 24);
                found_hash[i*4 + 1] = (hash[i] >> 16);
                found_hash[i*4 + 2] = (hash[i] >> 8);
                found_hash[i*4 + 3] = hash[i];
            }

            // Copy nonce from input buffer
            for (uint i = 0; i < nonce_len; i++) {
                found_nonce[i] = input[username_len + 1 + i];
            }

            // Record which thread found it
            *found_thread_idx = thread_idx;
        }
    }
}
