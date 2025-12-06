#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <utility>

namespace shallenge {

// RAII wrapper for cl_context
class CLContext {
    cl_context handle_ = nullptr;

public:
    CLContext() = default;
    explicit CLContext(cl_context handle) noexcept : handle_(handle) {}
    ~CLContext() { if (handle_) clReleaseContext(handle_); }

    CLContext(const CLContext&) = delete;
    CLContext& operator=(const CLContext&) = delete;

    CLContext(CLContext&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    CLContext& operator=(CLContext&& other) noexcept {
        if (this != &other) {
            if (handle_) clReleaseContext(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cl_context get() const noexcept { return handle_; }
    operator cl_context() const noexcept { return handle_; }
    [[nodiscard]] explicit operator bool() const noexcept { return handle_ != nullptr; }
};

// RAII wrapper for cl_command_queue
class CLCommandQueue {
    cl_command_queue handle_ = nullptr;

public:
    CLCommandQueue() = default;
    explicit CLCommandQueue(cl_command_queue handle) noexcept : handle_(handle) {}
    ~CLCommandQueue() { if (handle_) clReleaseCommandQueue(handle_); }

    CLCommandQueue(const CLCommandQueue&) = delete;
    CLCommandQueue& operator=(const CLCommandQueue&) = delete;

    CLCommandQueue(CLCommandQueue&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    CLCommandQueue& operator=(CLCommandQueue&& other) noexcept {
        if (this != &other) {
            if (handle_) clReleaseCommandQueue(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cl_command_queue get() const noexcept { return handle_; }
    operator cl_command_queue() const noexcept { return handle_; }
    [[nodiscard]] explicit operator bool() const noexcept { return handle_ != nullptr; }
};

// RAII wrapper for cl_program
class CLProgram {
    cl_program handle_ = nullptr;

public:
    CLProgram() = default;
    explicit CLProgram(cl_program handle) noexcept : handle_(handle) {}
    ~CLProgram() { if (handle_) clReleaseProgram(handle_); }

    CLProgram(const CLProgram&) = delete;
    CLProgram& operator=(const CLProgram&) = delete;

    CLProgram(CLProgram&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    CLProgram& operator=(CLProgram&& other) noexcept {
        if (this != &other) {
            if (handle_) clReleaseProgram(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cl_program get() const noexcept { return handle_; }
    operator cl_program() const noexcept { return handle_; }
    [[nodiscard]] explicit operator bool() const noexcept { return handle_ != nullptr; }
};

// RAII wrapper for cl_kernel
class CLKernel {
    cl_kernel handle_ = nullptr;

public:
    CLKernel() = default;
    explicit CLKernel(cl_kernel handle) noexcept : handle_(handle) {}
    ~CLKernel() { if (handle_) clReleaseKernel(handle_); }

    CLKernel(const CLKernel&) = delete;
    CLKernel& operator=(const CLKernel&) = delete;

    CLKernel(CLKernel&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    CLKernel& operator=(CLKernel&& other) noexcept {
        if (this != &other) {
            if (handle_) clReleaseKernel(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cl_kernel get() const noexcept { return handle_; }
    operator cl_kernel() const noexcept { return handle_; }
    [[nodiscard]] explicit operator bool() const noexcept { return handle_ != nullptr; }
};

// RAII wrapper for cl_mem (buffers)
class CLBuffer {
    cl_mem handle_ = nullptr;

public:
    CLBuffer() = default;
    explicit CLBuffer(cl_mem handle) noexcept : handle_(handle) {}
    ~CLBuffer() { if (handle_) clReleaseMemObject(handle_); }

    CLBuffer(const CLBuffer&) = delete;
    CLBuffer& operator=(const CLBuffer&) = delete;

    CLBuffer(CLBuffer&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    CLBuffer& operator=(CLBuffer&& other) noexcept {
        if (this != &other) {
            if (handle_) clReleaseMemObject(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cl_mem get() const noexcept { return handle_; }
    operator cl_mem() const noexcept { return handle_; }
    [[nodiscard]] explicit operator bool() const noexcept { return handle_ != nullptr; }
};

} // namespace shallenge
