#pragma once

#include "argmax_cuda.hpp"

#include "../../cuda/common.cuh"

namespace llaisys::ops::cuda {
namespace {
constexpr unsigned int BLOCK_SIZE = 256;
// Enough blocks to fill a large GPU while keeping the second stage to one block.
constexpr unsigned int MAX_BLOCKS = 256;
constexpr unsigned long long INVALID_INDEX = ~0ULL;

struct Candidate {
    float value;
    unsigned long long index;
};

__device__ __forceinline__ Candidate selectBetter(Candidate current, Candidate candidate) {
    if (candidate.index == INVALID_INDEX) {
        return current;
    }
    if (current.index == INVALID_INDEX) {
        return candidate;
    }

    const bool current_is_nan = current.value != current.value;
    const bool candidate_is_nan = candidate.value != candidate.value;
    if (candidate_is_nan != current_is_nan) {
        return candidate_is_nan ? candidate : current;
    }
    if (candidate_is_nan || candidate.value == current.value) {
        return candidate.index < current.index ? candidate : current;
    }
    return candidate.value > current.value ? candidate : current;
}

// Per-block winners handed from the first stage to the second. Sized statically so
// a reduction never has to cudaMalloc scratch on the inference path.
__device__ Candidate partials[MAX_BLOCKS];

__device__ __forceinline__ Candidate reduceBlock(Candidate candidate) {
    __shared__ Candidate candidates[BLOCK_SIZE];
    candidates[threadIdx.x] = candidate;
    __syncthreads();
    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            candidates[threadIdx.x] = selectBetter(
                candidates[threadIdx.x], candidates[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return candidates[0];
}

// Stage one: every block scans a strided slice of the input. Consecutive threads
// read consecutive elements, so the vocabulary is streamed coalesced across the
// whole GPU instead of by a single block.
template <typename T>
__global__ void argmaxPartialKernel(const T *values, size_t count) {
    Candidate candidate{0.0f, INVALID_INDEX};
    const size_t stride = static_cast<size_t>(gridDim.x) * BLOCK_SIZE;
    for (size_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < count; i += stride) {
        candidate = selectBetter(candidate,
                                 Candidate{toFloat(values[i]), static_cast<unsigned long long>(i)});
    }
    const Candidate winner = reduceBlock(candidate);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = winner;
    }
}

// Stage two: fold the per-block winners. `blocks` is at most MAX_BLOCKS, so one
// block finishes this in a single pass.
template <typename T>
__global__ void argmaxFinalKernel(int64_t *max_index, T *max_value, const T *values,
                                  unsigned int blocks) {
    Candidate candidate{0.0f, INVALID_INDEX};
    for (unsigned int i = threadIdx.x; i < blocks; i += BLOCK_SIZE) {
        candidate = selectBetter(candidate, partials[i]);
    }
    const Candidate winner = reduceBlock(candidate);
    if (threadIdx.x == 0) {
        max_index[0] = static_cast<int64_t>(winner.index);
        max_value[0] = values[winner.index];
    }
}

// Single-block path: one launch, no partials. Used when the input is small
// enough that a second kernel launch would cost more than the extra parallelism
// saves.
template <typename T>
__global__ void argmaxSingleKernel(int64_t *max_index, T *max_value, const T *values,
                                   size_t count) {
    Candidate candidate{0.0f, INVALID_INDEX};
    for (size_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
        candidate = selectBetter(candidate,
                                 Candidate{toFloat(values[i]), static_cast<unsigned long long>(i)});
    }
    const Candidate winner = reduceBlock(candidate);
    if (threadIdx.x == 0) {
        max_index[0] = static_cast<int64_t>(winner.index);
        max_value[0] = values[winner.index];
    }
}

template <typename T>
void launch(int64_t *max_index, std::byte *max_value, const std::byte *values, size_t count) {
    const auto *typed = reinterpret_cast<const T *>(values);
    const size_t needed = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Below this the kernel-launch overhead dominates the scan, so staying in a
    // single block is faster despite using one SM. Measured crossover on an
    // RTX 4090: ~8K elements, where both paths cost about 4 us.
    constexpr size_t single_block_limit = 32 * BLOCK_SIZE;
    if (count <= single_block_limit) {
        argmaxSingleKernel<<<1, BLOCK_SIZE>>>(max_index, reinterpret_cast<T *>(max_value),
                                              typed, count);
        checkKernel("argmax single kernel");
        return;
    }

    const unsigned int blocks = static_cast<unsigned int>(
        needed < MAX_BLOCKS ? needed : MAX_BLOCKS);
    argmaxPartialKernel<<<blocks, BLOCK_SIZE>>>(typed, count);
    checkKernel("argmax partial kernel");
    argmaxFinalKernel<<<1, BLOCK_SIZE>>>(max_index, reinterpret_cast<T *>(max_value),
                                         typed, blocks);
    checkKernel("argmax final kernel");
}
} // namespace

void argmax(int64_t *max_index, std::byte *max_value, const std::byte *values,
            llaisysDataType_t dtype, size_t count) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(max_index, max_value, values, count);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(max_index, max_value, values, count);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(max_index, max_value, values, count);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cuda
