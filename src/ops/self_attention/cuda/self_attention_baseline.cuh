#pragma once

// Pre-optimization decode attention, retained so the performance comparison in
// ASSIGNMENT_REPORT.md can be regenerated rather than merely described. Selected
// by `xmake f --baseline-kernels=y`; never compiled into a normal build.
//
// The shape of the cost is the point: one block per query head is 12 blocks for
// Qwen2-1.5B, so on a 128-SM GPU most of the device idles however efficient the
// block becomes, and each dot product is recomputed three times (once for the
// max pass, once for the sum, once for the output). Flash-decoding replaced it.
//
// This is a fragment, not a standalone header: it is included from inside the
// anonymous namespace of self_attention_cuda.cuh so that it can use that file's
// `attentionScore`, and it deliberately declares no namespaces of its own.

template <typename T>
__global__ void baselineDecodeAttentionKernel(T *out, const T *query, const T *key,
                                      const T *value, size_t kv_length,
                                      size_t query_heads, size_t kv_heads,
                                      size_t head_dimension, size_t value_dimension,
                                      float scale) {
    const size_t query_head = blockIdx.x;
    if (query_head >= query_heads) {
        return;
    }

    extern __shared__ float scratch[];
    __shared__ float maximum;
    __shared__ float denominator;

    const size_t kv_head = query_head / (query_heads / kv_heads);
    const size_t query_base = query_head * head_dimension;

    float local_max = -INFINITY;
    for (size_t key_position = threadIdx.x; key_position < kv_length;
         key_position += blockDim.x) {
        const size_t key_base = (key_position * kv_heads + kv_head) * head_dimension;
        local_max = fmaxf(
            local_max,
            attentionScore(query, key, query_base, key_base, head_dimension, scale));
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        maximum = scratch[0];
    }
    __syncthreads();

    float local_sum = 0.0F;
    for (size_t key_position = threadIdx.x; key_position < kv_length;
         key_position += blockDim.x) {
        const size_t key_base = (key_position * kv_heads + kv_head) * head_dimension;
        local_sum += expf(
            attentionScore(query, key, query_base, key_base, head_dimension, scale)
            - maximum);
    }
    scratch[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        denominator = scratch[0];
    }
    __syncthreads();

    float result = 0.0F;
    for (size_t chunk_start = 0; chunk_start < kv_length; chunk_start += blockDim.x) {
        const size_t key_position = chunk_start + threadIdx.x;
        if (key_position < kv_length) {
            const size_t key_base = (key_position * kv_heads + kv_head) * head_dimension;
            scratch[threadIdx.x] = expf(
                attentionScore(query, key, query_base, key_base, head_dimension, scale)
                - maximum)
                                 / denominator;
        }
        __syncthreads();

        if (threadIdx.x < value_dimension) {
            const size_t remaining = kv_length - chunk_start;
            const size_t chunk_size = remaining < blockDim.x ? remaining : blockDim.x;
            for (size_t offset = 0; offset < chunk_size; ++offset) {
                const size_t value_index =
                    ((chunk_start + offset) * kv_heads + kv_head) * value_dimension
                    + threadIdx.x;
                result += scratch[offset] * toFloat(value[value_index]);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < value_dimension) {
        out[query_head * value_dimension + threadIdx.x] = fromFloat<T>(result);
    }
}

template <typename T>
void launchBaselineDecode(std::byte *out, const std::byte *query, const std::byte *key,
                          const std::byte *value, size_t kv_length, size_t query_heads,
                          size_t kv_heads, size_t head_dimension, size_t value_dimension,
                          float scale) {
    constexpr int threads = 256;
    baselineDecodeAttentionKernel<<<static_cast<int>(query_heads), threads, threads * sizeof(float)>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(query),
        reinterpret_cast<const T *>(key), reinterpret_cast<const T *>(value),
        kv_length, query_heads, kv_heads, head_dimension, value_dimension, scale);
    checkKernel("decode attention kernel");
}
