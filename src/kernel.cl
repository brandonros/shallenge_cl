__kernel void vector_add(__global const float* a,
                         __global const float* b,
                         __global float* result) {
    int idx = get_global_id(0);
    result[idx] = a[idx] + b[idx];
}
