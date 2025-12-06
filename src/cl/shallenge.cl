// Shallenge mining kernel
// Note: This file is concatenated after sha256.cl and xoroshiro.cl

#define NONCE_LEN 21

__kernel void shallenge_mine(
    __global const uchar* username,      // e.g., "brandonros"
    uint username_len,                    // e.g., 10
    __global const uchar* target_hash,   // 32 bytes - current best to beat
    ulong rng_seed,                       // random seed from host
    __global uint* found_count,           // atomic counter for matches found
    __global uchar* found_hash,           // 32 bytes - best hash found
    __global uchar* found_nonce,          // 21 bytes - winning nonce
    __global uint* found_thread_idx       // which thread found it
) {
    size_t thread_idx = get_global_id(0);

    // Generate random nonce for this thread
    uchar nonce[NONCE_LEN];
    generate_base64_nonce(thread_idx, rng_seed, nonce, NONCE_LEN);

    // Build input: username + "/" + nonce
    // Total length should be exactly 32 bytes (e.g., 10 + 1 + 21 = 32)
    uchar input[32];

    // Copy username
    for (uint i = 0; i < username_len; i++) {
        input[i] = username[i];
    }

    // Add separator
    input[username_len] = '/';

    // Copy nonce
    for (uint i = 0; i < NONCE_LEN; i++) {
        input[username_len + 1 + i] = nonce[i];
    }

    // Compute SHA-256
    uchar hash[32];
    sha256_32(input, hash);

    // Compare to target (need to copy target to local memory for comparison)
    uchar target[32];
    for (int i = 0; i < 32; i++) {
        target[i] = target_hash[i];
    }

    // Check if this hash is better (lexicographically smaller)
    if (compare_hashes(hash, target) < 0) {
        // Found a better hash! Update outputs atomically

        // Increment found count
        atomic_inc(found_count);

        // Copy hash (race condition is acceptable - we just want any better hash)
        for (int i = 0; i < 32; i++) {
            found_hash[i] = hash[i];
        }

        // Copy nonce
        for (int i = 0; i < NONCE_LEN; i++) {
            found_nonce[i] = nonce[i];
        }

        // Record which thread found it
        *found_thread_idx = (uint)thread_idx;
    }
}
