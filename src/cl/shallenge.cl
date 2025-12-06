// Shallenge mining kernel
// Note: This file is concatenated after sha256.cl, util.cl, and nonce.cl

#define NONCE_LEN 21
#define HASHES_PER_THREAD 64

__kernel void shallenge_mine(
    __global const uchar* restrict username,      
    uint username_len,                             
    __global const uint* restrict target_hash,    // 8 uints (32 bytes as big-endian words)
    ulong rng_seed,                                // random seed from host
    __global uint* restrict found_count,           // atomic counter for matches found
    __global uchar* restrict found_hash,           // 32 bytes - best hash found
    __global uchar* restrict found_nonce,          // 21 bytes - winning nonce
    __global uint* restrict found_thread_idx,      // which thread found it
    __local uint* restrict target_local            // local memory for target hash (8 uints)
) {
    size_t thread_idx = get_global_id(0);
    int lid = get_local_id(0);

    // Load target hash into local memory (first 8 threads of each work-group)
    if (lid < 8) {
        target_local[lid] = target_hash[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Copy target from local to private memory for comparison (once per thread)
    uint target[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        target[i] = target_local[i];
    }

    // Initialize RNG state ONCE per thread (32-bit for GPU efficiency)
    uint s0, s1;
    init_rng_state(thread_idx, rng_seed, &s0, &s1);

    // Prepare input buffer with username prefix (doesn't change)
    uchar input[32];
    #pragma unroll
    for (uint i = 0; i < username_len; i++) {
        input[i] = username[i];
    }
    input[username_len] = '/';

    // Inner loop - hash multiple times per thread
    for (int iter = 0; iter < HASHES_PER_THREAD; iter++) {
        // Generate nonce using RNG state
        uchar nonce[NONCE_LEN];
        generate_nonce_from_state(&s0, &s1, nonce, NONCE_LEN);

        // Copy nonce to input
        #pragma unroll
        for (uint i = 0; i < NONCE_LEN; i++) {
            input[username_len + 1 + i] = nonce[i];
        }

        // Compute SHA-256
        uint hash[8];
        sha256_32_uint(input, hash);

        // Check if this hash is better
        if (is_hash_better(hash, target)) {
            atomic_inc(found_count);

            // Convert hash to bytes for output
            uchar hash_bytes[32];
            hash_uint_to_bytes(hash, hash_bytes);

            // Copy hash (race condition is acceptable)
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                found_hash[i] = hash_bytes[i];
            }

            // Copy nonce
            #pragma unroll
            for (int i = 0; i < NONCE_LEN; i++) {
                found_nonce[i] = nonce[i];
            }

            // Record which thread found it
            *found_thread_idx = (uint)thread_idx;
        }
    }
}
