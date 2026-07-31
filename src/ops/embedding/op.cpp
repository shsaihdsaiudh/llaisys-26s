#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#ifdef ENABLE_CUDA_COMPAT_OPS
#include "../cuda/dispatch.hpp"
#include "cuda/embedding_cuda.hpp"
#endif

#include <cstring>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding indices must use int64");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(index->ndim() == 1, "Embedding indices must be 1D");
    CHECK_ARGUMENT(weight->ndim() == 2 && out->ndim() == 2, "Embedding weight and output must be 2D");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0]
                       && out->shape()[1] == weight->shape()[1],
                   "Embedding output shape is invalid");
    CHECK_ARGUMENT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
                   "Embedding tensors must be contiguous");

    const auto *indices = reinterpret_cast<const int64_t *>(index->data());
#ifdef ENABLE_CUDA_COMPAT_OPS
    if (cuda::isAvailableDevice(out->deviceType())) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return cuda::embedding(out->data(), indices, weight->data(), out->dtype(),
                               index->numel(), weight->shape()[1], weight->shape()[0]);
    }
#endif
    CHECK_ARGUMENT(out->deviceType() == LLAISYS_DEVICE_CPU, "Unsupported embedding device");
    const size_t row_bytes = weight->shape()[1] * weight->elementSize();
    for (size_t i = 0; i < index->numel(); ++i) {
        CHECK_ARGUMENT(indices[i] >= 0 && static_cast<size_t>(indices[i]) < weight->shape()[0],
                       "Embedding index is out of range");
        std::memcpy(out->data() + i * row_bytes,
                    weight->data() + static_cast<size_t>(indices[i]) * row_bytes,
                    row_bytes);
    }
}
} // namespace llaisys::ops
