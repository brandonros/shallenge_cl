// Shallenge mining kernel
// Note: This file is concatenated after sha256.cl, util.cl, and nonce.cl

// HASHES_PER_THREAD is passed via -D at kernel compile time
// No fallback - kernel build will fail if not defined

// Max results per kernel launch - more than enough given hit probability
#define MAX_RESULTS 64

__kernel void shallenge_mine(
    __global const uchar* restrict username,
    uint username_len,
    __global const uint* restrict target_hash,    // 8 uints (32 bytes as big-endian words)
    uint rng_seed_lo,                              // Low 32 bits of 64-bit seed
    uint rng_seed_hi,                              // High 32 bits of 64-bit seed
    __global uint* restrict found_count,           // atomic counter - also used as slot allocator
    __global uchar* restrict found_hashes,         // [MAX_RESULTS * 32] bytes
    __global uchar* restrict found_nonces,         // [MAX_RESULTS * 32] bytes (padded for simplicity)
    __global uint* restrict found_thread_ids,      // [MAX_RESULTS] thread IDs for reproducibility
    __local uint* restrict target_local            // local memory for target hash (8 uints)
) {
    uint thread_idx = (uint)get_global_id(0);
    uint lid = get_local_id(0);

    // Calculate nonce length from username length (total input must be 32 bytes)
    uint nonce_len = 31 - username_len;  // 32 - username_len - 1 (for '/')

    // Load target hash into local memory (first 8 threads of each work-group)
    if (lid < 8) {
        target_local[lid] = target_hash[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Initialize RNG state ONCE per thread (full 64-bit entropy)
    uint s0, s1;
    init_rng_state(thread_idx, rng_seed_lo, rng_seed_hi, &s0, &s1);

    // Prepare input buffer with username prefix (doesn't change)
    uchar input[32];
    for (uint i = 0; i < username_len; i++) {
        input[i] = username[i];
    }
    input[username_len] = '/';

    // Inner loop - hash multiple times per thread
    for (int iter = 0; iter < HASHES_PER_THREAD; iter++) {
        generate_nonce_from_state(&s0, &s1, &input[username_len + 1], nonce_len);

        uint hash[8];
        sha256_32_uint(input, hash);

        if (is_hash_better(hash, target_local)) {
            // Atomically claim a slot
            uint slot = atomic_inc(found_count);

            if (slot < MAX_RESULTS) {
                // Write to our unique slot - no races possible
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
            // If slot >= MAX_RESULTS, we drop this result (extremely unlikely)
        }
    }
}
