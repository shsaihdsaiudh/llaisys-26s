#pragma once

#include "embedding_cuda.hpp"

#include "../../cuda/common.cuh"

namespace llaisys::ops::cuda {
namespace {
// One block per output row. The index is read once per row instead of once per
// element, and consecutive threads take consecutive columns so both the gather
// and the store coalesce.
//
// `weight_rows` guards the gather: the indices live in device memory, so
// validating them on the host would need a device-to-host sync on the decode
// path. An out-of-range or negative index would otherwise cast to a huge size_t
// and read out of bounds, so such rows are zero-filled instead.
template <typename T>
__global__ void embeddingKernel(T *out, const int64_t *indices, const T *weight,
                                size_t width, size_t weight_rows) {
    const size_t row = blockIdx.x;
    const int64_t index = indices[row];
    const bool valid = index >= 0 && static_cast<size_t>(index) < weight_rows;

    for (size_t column = threadIdx.x; column < width; column += blockDim.x) {
        out[row * width + column] = valid
                                      ? weight[static_cast<size_t>(index) * width + column]
                                      : fromFloat<T>(0.0F);
    }
}

template <typename T>
void launch(std::byte *out, const int64_t *indices, const std::byte *weight,
            size_t rows, size_t width, size_t weight_rows) {
    constexpr int threads = 256;
    embeddingKernel<<<static_cast<int>(rows), threads>>>(
        reinterpret_cast<T *>(out), indices,
        reinterpret_cast<const T *>(weight), width, weight_rows);
    checkKernel("embedding kernel");
}
} // namespace

void embedding(std::byte *out, const int64_t *indices, const std::byte *weight,
               llaisysDataType_t dtype, size_t rows, size_t width, size_t weight_rows) {
    if (rows == 0 || width == 0) {
        return;
    }
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, indices, weight, rows, width, weight_rows);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, indices, weight, rows, width, weight_rows);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, indices, weight, rows, width, weight_rows);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cuda
