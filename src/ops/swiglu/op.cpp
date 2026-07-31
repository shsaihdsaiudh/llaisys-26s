#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_CUDA_COMPAT_OPS
#include "../cuda/dispatch.hpp"
#include "cuda/swiglu_cuda.hpp"
#endif

#include <cmath>
#include <cstddef>

namespace {
template <typename T>
void swiglu_cpu(T *out, const T *gate, const T *up, size_t count) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t signed_index = 0; signed_index < static_cast<ptrdiff_t>(count); ++signed_index) {
        const size_t i = static_cast<size_t>(signed_index);
        const float gate_value = llaisys::utils::cast<float>(gate[i]);
        const float up_value = llaisys::utils::cast<float>(up[i]);
        const float silu = gate_value / (1.0F + std::exp(-gate_value));
        out[i] = llaisys::utils::cast<T>(up_value * silu);
    }
}
} // namespace

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    CHECK_ARGUMENT(out->ndim() == 2, "SwiGLU expects 2D tensors");
    CHECK_ARGUMENT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
                   "SwiGLU tensors must be contiguous");

#ifdef ENABLE_CUDA_COMPAT_OPS
    if (cuda::isAvailableDevice(out->deviceType())) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return cuda::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
    }
#endif
    CHECK_ARGUMENT(out->deviceType() == LLAISYS_DEVICE_CPU, "Unsupported SwiGLU device");

#define SWIGLU_CPU_CASE(DTYPE, TYPE)                                    \
    case DTYPE:                                                         \
        return swiglu_cpu(reinterpret_cast<TYPE *>(out->data()),        \
                          reinterpret_cast<const TYPE *>(gate->data()), \
                          reinterpret_cast<const TYPE *>(up->data()), out->numel())

    switch (out->dtype()) {
        SWIGLU_CPU_CASE(LLAISYS_DTYPE_F32, float);
        SWIGLU_CPU_CASE(LLAISYS_DTYPE_F16, fp16_t);
        SWIGLU_CPU_CASE(LLAISYS_DTYPE_BF16, bf16_t);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
#undef SWIGLU_CPU_CASE
}
} // namespace llaisys::ops
