#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_CUDA_COMPAT_OPS
#include "../cuda/dispatch.hpp"
#include "cuda/linear_cuda.hpp"
#endif

#include <cstddef>

namespace {
template <typename T>
void linear_cpu(T *out,
                const T *in,
                const T *weight,
                const T *bias,
                size_t rows,
                size_t in_features,
                size_t out_features) {
// MSVC implements OpenMP 2.0, which has no `collapse` and requires a signed
// loop index, so the two output dimensions are flattened by hand into one
// signed loop. Every OpenMP loop in the CPU kernels follows this shape.
#pragma omp parallel for schedule(static)
    for (ptrdiff_t flat = 0; flat < static_cast<ptrdiff_t>(rows * out_features); ++flat) {
        const size_t row = static_cast<size_t>(flat) / out_features;
        const size_t out_col = static_cast<size_t>(flat) % out_features;
        float sum = bias == nullptr ? 0.0F : llaisys::utils::cast<float>(bias[out_col]);
        for (size_t in_col = 0; in_col < in_features; ++in_col) {
            sum += llaisys::utils::cast<float>(in[row * in_features + in_col])
                 * llaisys::utils::cast<float>(weight[out_col * in_features + in_col]);
        }
        out[row * out_features + out_col] = llaisys::utils::cast<T>(sum);
    }
}
} // namespace

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias != nullptr) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2,
                   "Linear input, weight, and output must be 2D");
    CHECK_ARGUMENT(weight->shape()[1] == in->shape()[1], "Linear input feature count does not match weight");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0]
                       && out->shape()[1] == weight->shape()[0],
                   "Linear output shape is invalid");
    CHECK_ARGUMENT(bias == nullptr
                       || (bias->ndim() == 1 && bias->shape()[0] == weight->shape()[0]),
                   "Linear bias shape is invalid");
    CHECK_ARGUMENT(out->isContiguous() && in->isContiguous() && weight->isContiguous()
                       && (bias == nullptr || bias->isContiguous()),
                   "Linear tensors must be contiguous");

    const size_t rows = in->shape()[0];
    const size_t in_features = in->shape()[1];
    const size_t out_features = weight->shape()[0];

#ifdef ENABLE_CUDA_COMPAT_OPS
    if (cuda::isAvailableDevice(out->deviceType())) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return cuda::linear(out->data(), in->data(), weight->data(),
                            bias == nullptr ? nullptr : bias->data(), out->dtype(),
                            rows, in_features, out_features);
    }
#endif
    CHECK_ARGUMENT(out->deviceType() == LLAISYS_DEVICE_CPU, "Unsupported linear device");

#define LINEAR_CPU_CASE(DTYPE, TYPE)                                                                         \
    case DTYPE:                                                                                              \
        return linear_cpu(reinterpret_cast<TYPE *>(out->data()), reinterpret_cast<const TYPE *>(in->data()), \
                          reinterpret_cast<const TYPE *>(weight->data()),                                    \
                          bias == nullptr ? nullptr : reinterpret_cast<const TYPE *>(bias->data()),          \
                          rows, in_features, out_features)

    switch (out->dtype()) {
        LINEAR_CPU_CASE(LLAISYS_DTYPE_F32, float);
        LINEAR_CPU_CASE(LLAISYS_DTYPE_F16, fp16_t);
        LINEAR_CPU_CASE(LLAISYS_DTYPE_BF16, bf16_t);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
#undef LINEAR_CPU_CASE
}
} // namespace llaisys::ops
