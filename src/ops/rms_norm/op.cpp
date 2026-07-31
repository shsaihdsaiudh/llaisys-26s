#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_CUDA_COMPAT_OPS
#include "../cuda/dispatch.hpp"
#include "cuda/rms_norm_cuda.hpp"
#endif

#include <cmath>
#include <cstddef>

namespace {
template <typename T>
void rms_norm_cpu(T *out, const T *in, const T *weight, size_t rows, size_t width, float eps) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t signed_row = 0; signed_row < static_cast<ptrdiff_t>(rows); ++signed_row) {
        const size_t row = static_cast<size_t>(signed_row);
        float square_sum = 0.0F;
        for (size_t col = 0; col < width; ++col) {
            const float value = llaisys::utils::cast<float>(in[row * width + col]);
            square_sum += value * value;
        }
        const float scale = 1.0F / std::sqrt(square_sum / static_cast<float>(width) + eps);
        for (size_t col = 0; col < width; ++col) {
            const float value = llaisys::utils::cast<float>(in[row * width + col]);
            const float coefficient = llaisys::utils::cast<float>(weight[col]);
            out[row * width + col] = llaisys::utils::cast<T>(value * scale * coefficient);
        }
    }
}
} // namespace

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 1,
                   "RMS norm expects 2D input/output and 1D weight");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1], "RMS norm weight shape is invalid");
    CHECK_ARGUMENT(in->shape()[1] > 0, "RMS norm width must be non-zero");
    CHECK_ARGUMENT(eps >= 0.0F, "RMS norm epsilon must be non-negative");
    CHECK_ARGUMENT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
                   "RMS norm tensors must be contiguous");

#ifdef ENABLE_CUDA_COMPAT_OPS
    if (cuda::isAvailableDevice(out->deviceType())) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return cuda::rmsNorm(out->data(), in->data(), weight->data(), out->dtype(),
                             in->shape()[0], in->shape()[1], eps);
    }
#endif
    CHECK_ARGUMENT(out->deviceType() == LLAISYS_DEVICE_CPU, "Unsupported RMS norm device");

#define RMS_NORM_CPU_CASE(DTYPE, TYPE)                                                      \
    case DTYPE:                                                                             \
        return rms_norm_cpu(reinterpret_cast<TYPE *>(out->data()),                          \
                            reinterpret_cast<const TYPE *>(in->data()),                     \
                            reinterpret_cast<const TYPE *>(weight->data()), in->shape()[0], \
                            in->shape()[1], eps)

    switch (out->dtype()) {
        RMS_NORM_CPU_CASE(LLAISYS_DTYPE_F32, float);
        RMS_NORM_CPU_CASE(LLAISYS_DTYPE_F16, fp16_t);
        RMS_NORM_CPU_CASE(LLAISYS_DTYPE_BF16, bf16_t);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
#undef RMS_NORM_CPU_CASE
}
} // namespace llaisys::ops
