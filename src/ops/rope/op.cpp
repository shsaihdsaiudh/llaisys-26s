#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_CUDA_COMPAT_OPS
#include "../cuda/dispatch.hpp"
#include "cuda/rope_cuda.hpp"
#endif

#include <cmath>
#include <cstddef>
#include <vector>

namespace {
template <typename T>
void rope_cpu(T *out,
              const T *in,
              const int64_t *pos_ids,
              size_t seq_len,
              size_t heads,
              size_t head_dim,
              float theta) {
    const size_t half_dim = head_dim / 2;
    std::vector<float> frequency_divisor(half_dim);
    for (size_t i = 0; i < half_dim; ++i) {
        frequency_divisor[i] = std::pow(theta, 2.0F * static_cast<float>(i) / static_cast<float>(head_dim));
    }

#pragma omp parallel for schedule(static)
    for (ptrdiff_t flat = 0; flat < static_cast<ptrdiff_t>(seq_len * heads); ++flat) {
        const size_t token = static_cast<size_t>(flat) / heads;
        const size_t head = static_cast<size_t>(flat) % heads;
        const size_t base = (token * heads + head) * head_dim;
        for (size_t i = 0; i < half_dim; ++i) {
            const float angle = static_cast<float>(pos_ids[token]) / frequency_divisor[i];
            const float sine = std::sin(angle);
            const float cosine = std::cos(angle);
            const float a = llaisys::utils::cast<float>(in[base + i]);
            const float b = llaisys::utils::cast<float>(in[base + half_dim + i]);
            out[base + i] = llaisys::utils::cast<T>(a * cosine - b * sine);
            out[base + half_dim + i] = llaisys::utils::cast<T>(b * cosine + a * sine);
        }
    }
}
} // namespace

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(out->ndim() == 3, "RoPE input and output must be 3D");
    CHECK_ARGUMENT(pos_ids->ndim() == 1 && pos_ids->dtype() == LLAISYS_DTYPE_I64,
                   "RoPE position IDs must be a 1D int64 tensor");
    CHECK_ARGUMENT(pos_ids->shape()[0] == in->shape()[0], "RoPE position count must match sequence length");
    CHECK_ARGUMENT(in->shape()[2] > 0 && in->shape()[2] % 2 == 0,
                   "RoPE head dimension must be positive and even");
    CHECK_ARGUMENT(theta > 0.0F, "RoPE theta must be positive");
    CHECK_ARGUMENT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
                   "RoPE tensors must be contiguous");

#ifdef ENABLE_CUDA_COMPAT_OPS
    if (cuda::isAvailableDevice(out->deviceType())) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return cuda::rope(out->data(), in->data(),
                          reinterpret_cast<const int64_t *>(pos_ids->data()), out->dtype(),
                          in->shape()[0], in->shape()[1], in->shape()[2], theta);
    }
#endif
    CHECK_ARGUMENT(out->deviceType() == LLAISYS_DEVICE_CPU, "Unsupported RoPE device");

#define ROPE_CPU_CASE(DTYPE, TYPE)                                                          \
    case DTYPE:                                                                             \
        return rope_cpu(reinterpret_cast<TYPE *>(out->data()),                              \
                        reinterpret_cast<const TYPE *>(in->data()),                         \
                        reinterpret_cast<const int64_t *>(pos_ids->data()), in->shape()[0], \
                        in->shape()[1], in->shape()[2], theta)

    switch (out->dtype()) {
        ROPE_CPU_CASE(LLAISYS_DTYPE_F32, float);
        ROPE_CPU_CASE(LLAISYS_DTYPE_F16, fp16_t);
        ROPE_CPU_CASE(LLAISYS_DTYPE_BF16, bf16_t);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
#undef ROPE_CPU_CASE
}
} // namespace llaisys::ops
