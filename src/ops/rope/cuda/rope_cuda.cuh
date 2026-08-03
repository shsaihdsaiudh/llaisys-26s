#pragma once

#include "rope_cuda.hpp"

#include "../../cuda/common.cuh"

namespace llaisys::ops::cuda {
namespace {
#ifdef LLAISYS_BASELINE_KERNELS
// Pre-optimization kernel, retained only so the report's comparison can be
// regenerated. One thread per rotation pair, recomputing powf+sincosf per output
// element — `heads` times more transcendental work than the geometry requires.
template <typename T>
__global__ void ropeKernel(T *out, const T *input, const int64_t *positions,
                           size_t sequence_length, size_t heads,
                           size_t head_dimension, float theta) {
    const size_t half = head_dimension / 2;
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = sequence_length * heads * half;
    if (index >= count) {
        return;
    }
    const size_t pair_index = index % half;
    const size_t token_head = index / half;
    const size_t token = token_head / heads;
    const size_t base = token_head * head_dimension;
    const float divisor = powf(theta, 2.0F * static_cast<float>(pair_index)
                                          / static_cast<float>(head_dimension));
    const float angle = static_cast<float>(positions[token]) / divisor;
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    const float first = toFloat(input[base + pair_index]);
    const float second = toFloat(input[base + half + pair_index]);
    out[base + pair_index] = fromFloat<T>(first * cosine - second * sine);
    out[base + half + pair_index] = fromFloat<T>(second * cosine + first * sine);
}

template <typename T>
void launch(std::byte *out, const std::byte *input, const int64_t *positions,
            size_t sequence_length, size_t heads, size_t head_dimension, float theta) {
    constexpr int threads = 256;
    const size_t count = sequence_length * heads * (head_dimension / 2);
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    ropeKernel<<<blocks, threads>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(input),
                                    positions, sequence_length, heads, head_dimension, theta);
    checkKernel("rope kernel");
}
#else
// One block per token. The (sine, cosine) pair depends only on the token
// position and the rotation pair, not on the head, so each block computes
// head_dimension/2 of them once in shared memory and every head reuses them.
// The previous kernel recomputed powf+sincosf per output element, i.e. `heads`
// times more transcendental work than the geometry requires.
template <typename T>
__global__ void ropeKernel(T *out, const T *input, const int64_t *positions,
                           size_t heads, size_t head_dimension, float theta) {
    const size_t half = head_dimension / 2;
    const size_t token = blockIdx.x;
    const float position = static_cast<float>(positions[token]);

    extern __shared__ float trig[];
    float *sines = trig;
    float *cosines = trig + half;
    for (size_t pair = threadIdx.x; pair < half; pair += blockDim.x) {
        const float divisor = powf(theta, 2.0F * static_cast<float>(pair)
                                              / static_cast<float>(head_dimension));
        sincosf(position / divisor, &sines[pair], &cosines[pair]);
    }
    __syncthreads();

    // Consecutive threads take consecutive pair indices so the loads coalesce.
    for (size_t i = threadIdx.x; i < heads * half; i += blockDim.x) {
        const size_t pair = i % half;
        const size_t head = i / half;
        const size_t base = (token * heads + head) * head_dimension;
        const float first = toFloat(input[base + pair]);
        const float second = toFloat(input[base + half + pair]);
        const float sine = sines[pair];
        const float cosine = cosines[pair];
        out[base + pair] = fromFloat<T>(first * cosine - second * sine);
        out[base + half + pair] = fromFloat<T>(second * cosine + first * sine);
    }
}

template <typename T>
void launch(std::byte *out, const std::byte *input, const int64_t *positions,
            size_t sequence_length, size_t heads, size_t head_dimension, float theta) {
    constexpr int threads = 256;
    const size_t half = head_dimension / 2;
    ropeKernel<<<static_cast<int>(sequence_length), threads, 2 * half * sizeof(float)>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(input),
        positions, heads, head_dimension, theta);
    checkKernel("rope kernel");
}
#endif
} // namespace

void rope(std::byte *out, const std::byte *input, const int64_t *positions,
          llaisysDataType_t dtype, size_t sequence_length, size_t heads,
          size_t head_dimension, float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, input, positions, sequence_length, heads, head_dimension, theta);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, input, positions, sequence_length, heads, head_dimension, theta);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, input, positions, sequence_length, heads, head_dimension, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cuda
