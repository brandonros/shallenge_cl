#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "kernel.h"

int main() {
    cl_int err;

    // Get platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get platform" << std::endl;
        return 1;
    }

    // Get device
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get GPU device, trying CPU..." << std::endl;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to get any device" << std::endl;
            return 1;
        }
    }

    // Create context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context" << std::endl;
        return 1;
    }

    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue" << std::endl;
        return 1;
    }

    // Load and build kernel (embedded at compile time via xxd)
    const char* src = reinterpret_cast<const char*>(src_kernel_cl);
    size_t srcLen = src_kernel_cl_len;

    cl_program program = clCreateProgramWithSource(context, 1, &src, &srcLen, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program" << std::endl;
        return 1;
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build program" << std::endl;
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << log << std::endl;
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel" << std::endl;
        return 1;
    }

    // Prepare data
    const size_t N = 1024;
    std::vector<float> a(N), b(N), result(N);
    for (size_t i = 0; i < N; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    // Create buffers
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float) * N, a.data(), &err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float) * N, b.data(), &err);
    cl_mem bufResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                       sizeof(float) * N, nullptr, &err);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufResult);

    // Execute kernel
    size_t globalSize = N;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to enqueue kernel" << std::endl;
        return 1;
    }

    // Read results
    clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, sizeof(float) * N, result.data(), 0, nullptr, nullptr);

    // Verify and print
    std::cout << "Hello, OpenCL World!" << std::endl;
    std::cout << "Vector addition results (first 10):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << result[i] << std::endl;
    }

    // Cleanup
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
