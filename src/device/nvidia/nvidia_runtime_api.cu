#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace llaisys::device::nvidia {

namespace runtime_api {
namespace {
void checkCuda(cudaError_t status, const char *operation) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

// Teardown operations run from destructors, which may execute after the CUDA
// runtime has begun unloading at process exit. `cudaErrorCudartUnloading` means
// the runtime is already gone, so the resource this call would have released no
// longer exists and the call is a successful no-op. Any other status is a real
// failure and still throws.
void checkCudaTeardown(cudaError_t status, const char *operation) {
    if (status == cudaErrorCudartUnloading) {
        return;
    }
    checkCuda(status, operation);
}

cudaMemcpyKind memcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("Unsupported CUDA memcpy kind");
    }
}
} // namespace

int getDeviceCount() {
    int count = 0;
    checkCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
}

void setDevice(int device) {
    // Also reached from Runtime's destructor at process exit.
    checkCudaTeardown(cudaSetDevice(device), "cudaSetDevice");
}

void deviceSynchronize() {
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        checkCudaTeardown(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamDestroy");
    }
}
void streamSynchronize(llaisysStream_t stream) {
    checkCuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
}

void *mallocDevice(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    checkCuda(cudaMalloc(&ptr, size), "cudaMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        checkCudaTeardown(cudaFree(ptr), "cudaFree");
    }
}

void *mallocHost(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    checkCuda(cudaMallocHost(&ptr, size), "cudaMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        checkCudaTeardown(cudaFreeHost(ptr), "cudaFreeHost");
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    if (size != 0) {
        checkCuda(cudaMemcpy(dst, src, size, memcpyKind(kind)), "cudaMemcpy");
    }
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    if (size != 0) {
        checkCuda(cudaMemcpyAsync(dst, src, size, memcpyKind(kind), reinterpret_cast<cudaStream_t>(stream)),
                  "cudaMemcpyAsync");
    }
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
