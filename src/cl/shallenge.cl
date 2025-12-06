// Shallenge mining kernel
// Note: This file is concatenated after sha256.cl and xoroshiro.cl

#define NONCE_LEN 21

__kernel void shallenge_mine(
    __global const uchar* restrict username,      // e.g., "brandonros"
    uint username_len,                             // e.g., 10
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

    // Generate random nonce for this thread
    uchar nonce[NONCE_LEN];
    generate_base64_nonce(thread_idx, rng_seed, nonce, NONCE_LEN);

    // Build input: username + "/" + nonce (32 bytes total)
    uchar input[32];

    // Copy username (unrolled for common case of 10 chars)
    #pragma unroll
    for (uint i = 0; i < username_len; i++) {
        input[i] = username[i];
    }

    // Add separator
    input[username_len] = '/';

    // Copy nonce
    #pragma unroll
    for (uint i = 0; i < NONCE_LEN; i++) {
        input[username_len + 1 + i] = nonce[i];
    }

    // Compute SHA-256 (returns uint[8] for efficient comparison)
    uint hash[8];
    sha256_32_uint(input, hash);

    // Copy target from local to private memory for comparison
    uint target[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        target[i] = target_local[i];
    }

    // Check if this hash is better using early-exit comparison
    if (is_hash_better(hash, target)) {
        // Found a better hash! Update outputs atomically
        atomic_inc(found_count);

        // Convert hash to bytes for output (only done for winners)
        uchar hash_bytes[32];
        hash_uint_to_bytes(hash, hash_bytes);

        // Copy hash (race condition is acceptable - we just want any better hash)
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
