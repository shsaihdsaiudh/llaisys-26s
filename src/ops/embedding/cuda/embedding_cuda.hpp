#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cuda {
void embedding(std::byte *out, const int64_t *indices, const std::byte *weight,
               llaisysDataType_t dtype, size_t rows, size_t width, size_t weight_rows);
}
