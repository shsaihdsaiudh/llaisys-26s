#pragma once

#include "self_attention_cuda.hpp"

#include "../../cuda/common.cuh"

#include <cmath>

namespace llaisys::ops::cuda {
namespace {
template <typename T>
__device__ float attentionScore(const T *query, const T *key,
                                size_t query_base, size_t key_base,
                                size_t head_dimension, float scale) {
    float result = 0.0F;
    for (size_t column = 0; column < head_dimension; ++column) {
        result += toFloat(query[query_base + column]) * toFloat(key[key_base + column]);
    }
    return result * scale;
}

template <typename T>
__global__ void attentionScoresKernel(float *scores, const T *query, const T *key,
                                      size_t query_length, size_t kv_length,
                                      size_t query_heads, size_t kv_heads,
                                      size_t head_dimension, float scale) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = query_length * query_heads * kv_length;
    if (index >= count) {
        return;
    }
    const size_t key_position = index % kv_length;
    const size_t query_head_index = index / kv_length;
    const size_t query_head = query_head_index % query_heads;
    const size_t query_position = query_head_index / query_heads;
    const size_t cached_tokens = kv_length - query_length;
    if (key_position >= cached_tokens + query_position + 1) {
        scores[index] = -INFINITY;
        return;
    }
    const size_t kv_head = query_head / (query_heads / kv_heads);
    const size_t query_base = (query_position * query_heads + query_head) * head_dimension;
    const size_t key_base = (key_position * kv_heads + kv_head) * head_dimension;
    scores[index] = attentionScore(query, key, query_base, key_base, head_dimension, scale);
}

__global__ void softmaxKernel(float *scores, size_t rows, size_t columns) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    extern __shared__ float scratch[];
    float local_max = -INFINITY;
    for (size_t column = threadIdx.x; column < columns; column += blockDim.x) {
        local_max = fmaxf(local_max, scores[row * columns + column]);
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float maximum = scratch[0];
    float local_sum = 0.0F;
    for (size_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const float probability = expf(scores[row * columns + column] - maximum);
        scores[row * columns + column] = probability;
        local_sum += probability;
    }
    scratch[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denominator = scratch[0];
    for (size_t column = threadIdx.x; column < columns; column += blockDim.x) {
        scores[row * columns + column] /= denominator;
    }
}

template <typename T>
__global__ void attentionValuesKernel(T *out, const float *scores, const T *value,
                                      size_t query_length, size_t kv_length,
                                      size_t query_heads, size_t kv_heads,
                                      size_t value_dimension) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = query_length * query_heads * value_dimension;
    if (index >= count) {
        return;
    }
    const size_t column = index % value_dimension;
    const size_t query_head_index = index / value_dimension;
    const size_t query_head = query_head_index % query_heads;
    const size_t kv_head = query_head / (query_heads / kv_heads);
    float result = 0.0F;
    for (size_t key_position = 0; key_position < kv_length; ++key_position) {
        const float probability = scores[query_head_index * kv_length + key_position];
        const size_t value_index = (key_position * kv_heads + kv_head) * value_dimension + column;
        result += probability * toFloat(value[value_index]);
    }
    out[index] = fromFloat<T>(result);
}

#ifdef LLAISYS_BASELINE_KERNELS
// Brings in launchBaselineDecode. Included here, inside the anonymous namespace,
// because the fragment uses attentionScore above.
#include "self_attention_baseline.cuh"
#endif

// Sums `value` across the 32 lanes of a warp without any shared memory or
// barriers. Used instead of a shared-memory tree so the per-key dot products do
// not pay a __syncthreads each.
__device__ __forceinline__ float warpSum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

// Flash-decoding split-KV.
//
// Decoding a single token gives only `query_heads` independent units of work
// (12 for Qwen2-1.5B), so a block-per-head kernel leaves most of a 128-SM GPU
// idle and tops out around 7 GB/s no matter how long the context is. Splitting
// the key range into chunks makes the grid query_heads * splits, and each chunk
// is combined afterwards using the standard log-sum-exp merge, so the result is
// identical to a single-pass softmax.
//
// Each block writes its chunk's softmax statistics (max, sum) and its
// unnormalized weighted V.
template <typename T>
__global__ void decodeSplitKernel(float *partial_out, float *partial_max,
                                  float *partial_sum, const T *query, const T *key,
                                  const T *value, size_t kv_length, size_t query_heads,
                                  size_t kv_heads, size_t head_dimension,
                                  size_t value_dimension, float scale, size_t chunk) {
    const size_t query_head = blockIdx.x;
    const size_t split = blockIdx.y;
    const size_t begin = split * chunk;
    if (query_head >= query_heads || begin >= kv_length) {
        return;
    }
    const size_t end = min(begin + chunk, kv_length);
    const size_t kv_head = query_head / (query_heads / kv_heads);

    const unsigned int warps = blockDim.y;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const size_t threads = blockDim.x * warps;
    const size_t flat = warp * blockDim.x + lane;

    extern __shared__ float shared[];
    float *staged_query = shared;                     // head_dimension
    float *scores = staged_query + head_dimension;    // chunk
    float *scratch = scores + chunk;                  // threads

    for (size_t column = flat; column < head_dimension; column += threads) {
        staged_query[column] = toFloat(query[query_head * head_dimension + column]);
    }
    __syncthreads();

    for (size_t position = begin + warp; position < end; position += warps) {
        const size_t key_base = (position * kv_heads + kv_head) * head_dimension;
        float partial = 0.0F;
        for (size_t column = lane; column < head_dimension; column += blockDim.x) {
            partial += staged_query[column] * toFloat(key[key_base + column]);
        }
        partial = warpSum(partial);
        if (lane == 0) {
            scores[position - begin] = partial * scale;
        }
    }
    __syncthreads();

    const size_t span = end - begin;
    float local_max = -INFINITY;
    for (size_t i = flat; i < span; i += threads) {
        local_max = fmaxf(local_max, scores[i]);
    }
    scratch[flat] = local_max;
    __syncthreads();
    for (unsigned int stride = threads / 2; stride > 0; stride /= 2) {
        if (flat < stride) {
            scratch[flat] = fmaxf(scratch[flat], scratch[flat + stride]);
        }
        __syncthreads();
    }
    const float maximum = scratch[0];
    __syncthreads();

    float local_sum = 0.0F;
    for (size_t i = flat; i < span; i += threads) {
        const float probability = expf(scores[i] - maximum);
        scores[i] = probability;
        local_sum += probability;
    }
    scratch[flat] = local_sum;
    __syncthreads();
    for (unsigned int stride = threads / 2; stride > 0; stride /= 2) {
        if (flat < stride) {
            scratch[flat] += scratch[flat + stride];
        }
        __syncthreads();
    }
    const float denominator = scratch[0];
    __syncthreads();

    const size_t slot = query_head * gridDim.y + split;
    if (flat == 0) {
        partial_max[slot] = maximum;
        partial_sum[slot] = denominator;
    }

    // Unnormalized so the combine step can rescale with the global maximum.
    for (size_t column = lane; column < value_dimension; column += blockDim.x) {
        float accumulated = 0.0F;
        for (size_t position = begin + warp; position < end; position += warps) {
            accumulated += scores[position - begin]
                         * toFloat(value[(position * kv_heads + kv_head) * value_dimension + column]);
        }
        scratch[flat] = accumulated;
        __syncthreads();
        for (unsigned int stride = warps / 2; stride > 0; stride /= 2) {
            if (warp < stride) {
                scratch[flat] += scratch[(warp + stride) * blockDim.x + lane];
            }
            __syncthreads();
        }
        if (warp == 0) {
            partial_out[slot * value_dimension + column] = scratch[lane];
        }
        __syncthreads();
    }
}

// Merges the per-chunk results. One block per query head; `splits` is small.
template <typename T>
__global__ void decodeCombineKernel(T *out, const float *partial_out,
                                    const float *partial_max, const float *partial_sum,
                                    size_t value_dimension, unsigned int splits) {
    const size_t query_head = blockIdx.x;
    const size_t base = query_head * splits;

    extern __shared__ float weights[];
    float maximum = -INFINITY;
    for (unsigned int split = 0; split < splits; ++split) {
        maximum = fmaxf(maximum, partial_max[base + split]);
    }
    float denominator = 0.0F;
    for (unsigned int split = 0; split < splits; ++split) {
        const float weight = expf(partial_max[base + split] - maximum);
        if (threadIdx.x == 0) {
            weights[split] = weight;
        }
        denominator += partial_sum[base + split] * weight;
    }
    __syncthreads();

    for (size_t column = threadIdx.x; column < value_dimension; column += blockDim.x) {
        float total = 0.0F;
        for (unsigned int split = 0; split < splits; ++split) {
            total += partial_out[(base + split) * value_dimension + column] * weights[split];
        }
        out[query_head * value_dimension + column] = fromFloat<T>(total / denominator);
    }
}

template <typename T>
__global__ void decodeAttentionKernel(T *out, const T *query, const T *key,
                                      const T *value, size_t kv_length,
                                      size_t query_heads, size_t kv_heads,
                                      size_t head_dimension, size_t value_dimension,
                                      float scale) {
    const size_t query_head = blockIdx.x;
    if (query_head >= query_heads) {
        return;
    }
    const size_t kv_head = query_head / (query_heads / kv_heads);
    const unsigned int warps = blockDim.y;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const size_t threads = blockDim.x * blockDim.y;
    const size_t flat = warp * blockDim.x + lane;

    extern __shared__ float shared[];
    float *staged_query = shared;                     // head_dimension
    float *scores = staged_query + head_dimension;    // kv_length
    float *scratch = scores + kv_length;              // threads

    // Stage the query row once instead of re-reading it from global memory for
    // every key position.
    for (size_t column = flat; column < head_dimension; column += threads) {
        staged_query[column] = toFloat(query[query_head * head_dimension + column]);
    }
    __syncthreads();

    // One score per key position, computed exactly once and kept in shared
    // memory. Each warp owns a key; lanes stride head_dimension so their reads
    // coalesce, then the warp reduces via shuffle with no barrier.
    for (size_t key_position = warp; key_position < kv_length; key_position += warps) {
        const size_t key_base = (key_position * kv_heads + kv_head) * head_dimension;
        float partial = 0.0F;
        for (size_t column = lane; column < head_dimension; column += blockDim.x) {
            partial += staged_query[column] * toFloat(key[key_base + column]);
        }
        partial = warpSum(partial);
        if (lane == 0) {
            scores[key_position] = partial * scale;
        }
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (size_t i = flat; i < kv_length; i += threads) {
        local_max = fmaxf(local_max, scores[i]);
    }
    scratch[flat] = local_max;
    __syncthreads();
    for (unsigned int stride = threads / 2; stride > 0; stride /= 2) {
        if (flat < stride) {
            scratch[flat] = fmaxf(scratch[flat], scratch[flat + stride]);
        }
        __syncthreads();
    }
    const float maximum = scratch[0];
    __syncthreads();

    float local_sum = 0.0F;
    for (size_t i = flat; i < kv_length; i += threads) {
        const float probability = expf(scores[i] - maximum);
        scores[i] = probability;
        local_sum += probability;
    }
    scratch[flat] = local_sum;
    __syncthreads();
    for (unsigned int stride = threads / 2; stride > 0; stride /= 2) {
        if (flat < stride) {
            scratch[flat] += scratch[flat + stride];
        }
        __syncthreads();
    }
    const float denominator = scratch[0];
    __syncthreads();

    // Weighted sum of V. Lanes take consecutive columns so the reads coalesce,
    // and the warps split the key range so long contexts stay parallel; each
    // warp's partial is then folded through shared memory.
    for (size_t column = lane; column < value_dimension; column += blockDim.x) {
        float partial = 0.0F;
        for (size_t key_position = warp; key_position < kv_length; key_position += warps) {
            partial += scores[key_position]
                     * toFloat(value[(key_position * kv_heads + kv_head) * value_dimension + column]);
        }
        scratch[flat] = partial;
        __syncthreads();
        for (unsigned int stride = warps / 2; stride > 0; stride /= 2) {
            if (warp < stride) {
                scratch[flat] += scratch[(warp + stride) * blockDim.x + lane];
            }
            __syncthreads();
        }
        if (warp == 0) {
            out[query_head * value_dimension + column] = fromFloat<T>(scratch[lane] / denominator);
        }
        __syncthreads();
    }
}

// Shared memory the fused decode path needs for a given geometry.
inline size_t decodeSharedBytes(size_t kv_length, size_t head_dimension, size_t threads) {
    return (head_dimension + kv_length + threads) * sizeof(float);
}

// Scratch for the split-KV path, grown on demand and reused across calls so the
// decode loop never allocates. Freed when the process exits.
struct SplitWorkspace {
    float *memory = nullptr;
    size_t floats = 0;

    float *reserve(size_t needed) {
        if (needed <= floats) {
            return memory;
        }
        if (memory != nullptr) {
            cudaFree(memory);
            memory = nullptr;
            floats = 0;
        }
        checkCuda(cudaMalloc(&memory, needed * sizeof(float)), "cudaMalloc attention split workspace");
        floats = needed;
        return memory;
    }
};

inline SplitWorkspace &splitWorkspace() {
    thread_local SplitWorkspace workspace;
    return workspace;
}

template <typename T>
void launchDecodeSplit(std::byte *out, const std::byte *query, const std::byte *key,
                       const std::byte *value, size_t kv_length, size_t query_heads,
                       size_t kv_heads, size_t head_dimension, size_t value_dimension,
                       float scale, size_t chunk) {
    const dim3 threads(32, 8);
    const unsigned int total = threads.x * threads.y;
    const auto splits = static_cast<unsigned int>((kv_length + chunk - 1) / chunk);

    // partial V, then the per-chunk max and sum.
    const size_t slots = query_heads * splits;
    const size_t needed = slots * value_dimension + 2 * slots;
    float *scratch = splitWorkspace().reserve(needed);
    float *partial_out = scratch;
    float *partial_max = partial_out + slots * value_dimension;
    float *partial_sum = partial_max + slots;

    const dim3 grid(static_cast<unsigned int>(query_heads), splits);
    decodeSplitKernel<<<grid, threads,
                        (head_dimension + chunk + total) * sizeof(float)>>>(
        partial_out, partial_max, partial_sum, reinterpret_cast<const T *>(query),
        reinterpret_cast<const T *>(key), reinterpret_cast<const T *>(value),
        kv_length, query_heads, kv_heads, head_dimension, value_dimension, scale, chunk);
    checkKernel("decode attention split kernel");

    decodeCombineKernel<T><<<static_cast<unsigned int>(query_heads), 128,
                             splits * sizeof(float)>>>(
        reinterpret_cast<T *>(out), partial_out, partial_max, partial_sum,
        value_dimension, splits);
    checkKernel("decode attention combine kernel");
}

template <typename T>
void launchDecode(std::byte *out, const std::byte *query, const std::byte *key,
                  const std::byte *value, size_t kv_length, size_t query_heads,
                  size_t kv_heads, size_t head_dimension, size_t value_dimension,
                  float scale) {
    // threads.x reduces one dot product, threads.y processes that many keys at
    // once. Both are powers of two so the tree reductions stay exact.
    const dim3 threads(32, 8);
    const size_t total = threads.x * threads.y;
    const size_t shared = decodeSharedBytes(kv_length, head_dimension, total);
    decodeAttentionKernel<<<static_cast<int>(query_heads), threads, shared>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(query),
        reinterpret_cast<const T *>(key), reinterpret_cast<const T *>(value),
        kv_length, query_heads, kv_heads, head_dimension, value_dimension, scale);
    checkKernel("decode attention kernel");
}

template <typename T>
void launch(std::byte *out, const std::byte *query, const std::byte *key,
            const std::byte *value, size_t query_length, size_t kv_length,
            size_t query_heads, size_t kv_heads, size_t head_dimension,
            size_t value_dimension, float scale) {
    if (query_length == 1) {
#ifdef LLAISYS_BASELINE_KERNELS
        // One block per query head at every context length — the geometry that
        // flash-decoding replaced. See self_attention_baseline.cuh.
        return launchBaselineDecode<T>(out, query, key, value, kv_length, query_heads,
                                       kv_heads, head_dimension, value_dimension, scale);
#else
        // A block per query head is only 12 blocks for Qwen2-1.5B, so once the
        // context is long enough to matter, split the key range to fill the GPU.
        // Below that the split's combine pass costs more than it saves.
        constexpr size_t split_threshold = 1024;
        constexpr size_t chunk = 512;
        if (kv_length > split_threshold) {
            return launchDecodeSplit<T>(out, query, key, value, kv_length, query_heads,
                                        kv_heads, head_dimension, value_dimension,
                                        scale, chunk);
        }
        return launchDecode<T>(out, query, key, value, kv_length, query_heads,
                               kv_heads, head_dimension, value_dimension, scale);
#endif
    }

    constexpr int threads = 256;
    const size_t score_count = query_length * query_heads * kv_length;
    float *scores = nullptr;
    checkCuda(cudaMalloc(&scores, score_count * sizeof(float)), "cudaMalloc attention scores");
    try {
        const int score_blocks = static_cast<int>((score_count + threads - 1) / threads);
        attentionScoresKernel<<<score_blocks, threads>>>(
            scores, reinterpret_cast<const T *>(query), reinterpret_cast<const T *>(key),
            query_length, kv_length, query_heads, kv_heads, head_dimension, scale);
        checkKernel("attention scores kernel");

        const size_t score_rows = query_length * query_heads;
        softmaxKernel<<<static_cast<int>(score_rows), threads, threads * sizeof(float)>>>(
            scores, score_rows, kv_length);
        checkKernel("attention softmax kernel");

        const size_t output_count = query_length * query_heads * value_dimension;
        const int output_blocks = static_cast<int>((output_count + threads - 1) / threads);
        attentionValuesKernel<<<output_blocks, threads>>>(
            reinterpret_cast<T *>(out), scores, reinterpret_cast<const T *>(value),
            query_length, kv_length, query_heads, kv_heads, value_dimension);
        checkKernel("attention values kernel");
        checkCuda(cudaFree(scores), "cudaFree attention scores");
    } catch (...) {
        cudaFree(scores);
        throw;
    }
}
} // namespace

void selfAttention(std::byte *out, const std::byte *query, const std::byte *key,
                   const std::byte *value, llaisysDataType_t dtype,
                   size_t query_length, size_t kv_length, size_t query_heads,
                   size_t kv_heads, size_t head_dimension, size_t value_dimension,
                   float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, query, key, value, query_length, kv_length,
                             query_heads, kv_heads, head_dimension, value_dimension, scale);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, query, key, value, query_length, kv_length,
                              query_heads, kv_heads, head_dimension, value_dimension, scale);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, query, key, value, query_length, kv_length,
                                     query_heads, kv_heads, head_dimension, value_dimension, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cuda
