#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_CUDA_COMPAT_OPS
#include "../cuda/dispatch.hpp"
#include "cuda/self_attention_cuda.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {
template <typename T>
void self_attention_cpu(T *attn_val,
                        const T *q,
                        const T *k,
                        const T *v,
                        size_t query_len,
                        size_t kv_len,
                        size_t query_heads,
                        size_t kv_heads,
                        size_t head_dim,
                        size_t value_dim,
                        float scale) {
    const size_t queries_per_kv_head = query_heads / kv_heads;
    const size_t cached_tokens = kv_len - query_len;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t flat = 0; flat < static_cast<ptrdiff_t>(query_len * query_heads); ++flat) {
        const size_t query_pos = static_cast<size_t>(flat) / query_heads;
        const size_t query_head = static_cast<size_t>(flat) % query_heads;
        const size_t kv_head = query_head / queries_per_kv_head;
        const size_t visible_keys = cached_tokens + query_pos + 1;
        std::vector<float> scores(visible_keys);
        float max_score = -std::numeric_limits<float>::infinity();

        const size_t query_base = (query_pos * query_heads + query_head) * head_dim;
        for (size_t key_pos = 0; key_pos < visible_keys; ++key_pos) {
            const size_t key_base = (key_pos * kv_heads + kv_head) * head_dim;
            float score = 0.0F;
            for (size_t col = 0; col < head_dim; ++col) {
                score += llaisys::utils::cast<float>(q[query_base + col])
                       * llaisys::utils::cast<float>(k[key_base + col]);
            }
            score *= scale;
            scores[key_pos] = score;
            max_score = std::max(max_score, score);
        }

        float exp_sum = 0.0F;
        for (float &score : scores) {
            score = std::exp(score - max_score);
            exp_sum += score;
        }

        const size_t output_base = (query_pos * query_heads + query_head) * value_dim;
        for (size_t col = 0; col < value_dim; ++col) {
            float result = 0.0F;
            for (size_t key_pos = 0; key_pos < visible_keys; ++key_pos) {
                const size_t value_base = (key_pos * kv_heads + kv_head) * value_dim;
                result += scores[key_pos] * llaisys::utils::cast<float>(v[value_base + col]);
            }
            attn_val[output_base + col] = llaisys::utils::cast<T>(result / exp_sum);
        }
    }
}
} // namespace

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
                   "Self attention tensors must be 3D");
    CHECK_ARGUMENT(q->shape()[0] > 0 && q->shape()[0] <= k->shape()[0],
                   "Self attention query length must not exceed key length");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0] && k->shape()[1] == v->shape()[1],
                   "Self attention key and value shapes are incompatible");
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2],
                   "Self attention query and key dimensions must match");
    CHECK_ARGUMENT(k->shape()[1] > 0 && q->shape()[1] % k->shape()[1] == 0,
                   "Self attention query heads must be divisible by KV heads");
    CHECK_ARGUMENT(attn_val->shape()[0] == q->shape()[0]
                       && attn_val->shape()[1] == q->shape()[1]
                       && attn_val->shape()[2] == v->shape()[2],
                   "Self attention output shape is invalid");
    CHECK_ARGUMENT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous()
                       && v->isContiguous(),
                   "Self attention tensors must be contiguous");

#ifdef ENABLE_CUDA_COMPAT_OPS
    if (cuda::isAvailableDevice(attn_val->deviceType())) {
        llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
        return cuda::selfAttention(
            attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(),
            q->shape()[0], k->shape()[0], q->shape()[1], k->shape()[1],
            q->shape()[2], v->shape()[2], scale);
    }
#endif
    CHECK_ARGUMENT(attn_val->deviceType() == LLAISYS_DEVICE_CPU, "Unsupported self attention device");

#define SELF_ATTENTION_CPU_CASE(DTYPE, TYPE)                                                       \
    case DTYPE:                                                                                    \
        return self_attention_cpu(                                                                 \
            reinterpret_cast<TYPE *>(attn_val->data()), reinterpret_cast<const TYPE *>(q->data()), \
            reinterpret_cast<const TYPE *>(k->data()), reinterpret_cast<const TYPE *>(v->data()),  \
            q->shape()[0], k->shape()[0], q->shape()[1], k->shape()[1], q->shape()[2],             \
            v->shape()[2], scale)

    switch (attn_val->dtype()) {
        SELF_ATTENTION_CPU_CASE(LLAISYS_DTYPE_F32, float);
        SELF_ATTENTION_CPU_CASE(LLAISYS_DTYPE_F16, fp16_t);
        SELF_ATTENTION_CPU_CASE(LLAISYS_DTYPE_BF16, bf16_t);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
    }
#undef SELF_ATTENTION_CPU_CASE
}
} // namespace llaisys::ops
