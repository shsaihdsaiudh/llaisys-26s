import os
import sys
import ctypes
from pathlib import Path

from .runtime import load_runtime
from .runtime import LlaisysRuntimeAPI
from .error import LlaisysError, load_error, install_error_checks
from .llaisys_types import llaisysDeviceType_t, DeviceType
from .llaisys_types import llaisysDataType_t, DataType
from .llaisys_types import llaisysMemcpyKind_t, MemcpyKind
from .llaisys_types import llaisysStream_t
from .tensor import llaisysTensor_t
from .tensor import load_tensor
from .ops import load_ops
from .qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights, llaisysQwen2Model_t
from .qwen2 import load_qwen2


def load_shared_library():
    lib_dir = Path(__file__).parent

    if sys.platform.startswith("linux"):
        libname = "libllaisys.so"
    elif sys.platform == "win32":
        libname = "llaisys.dll"
    elif sys.platform == "darwin":
        libname = "llaisys.dylib"
    else:
        raise RuntimeError("Unsupported platform")

    lib_path = os.path.join(lib_dir, libname)

    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Shared library not found: {lib_path}")

    return ctypes.CDLL(str(lib_path))


LIB_LLAISYS = load_shared_library()
load_error(LIB_LLAISYS)
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)
load_qwen2(LIB_LLAISYS)

# Backend calls report failures through llaisysGetLastError(); raise instead.
install_error_checks(
    LIB_LLAISYS,
    [
        "llaisysAdd",
        "llaisysArgmax",
        "llaisysEmbedding",
        "llaisysGetRuntimeAPI",
        "llaisysLinear",
        "llaisysQwen2ModelCreate",
        "llaisysQwen2ModelDestroy",
        "llaisysQwen2ModelInfer",
        "llaisysQwen2ModelReserveCache",
        "llaisysQwen2ModelReset",
        "llaisysQwen2ModelWeights",
        "llaisysROPE",
        "llaisysRearrange",
        "llaisysRmsNorm",
        "llaisysSelfAttention",
        "llaisysSetContextRuntime",
        "llaisysSwiGLU",
        "tensorCreate",
        "tensorDebug",
        "tensorDestroy",
        "tensorGetData",
        "tensorGetDataType",
        "tensorGetDeviceId",
        "tensorGetDeviceType",
        "tensorGetNdim",
        "tensorGetShape",
        "tensorGetStrides",
        "tensorIsContiguous",
        "tensorLoad",
        "tensorPermute",
        "tensorSlice",
        "tensorView",
    ],
)


__all__ = [
    "LIB_LLAISYS",
    "LlaisysError",
    "LlaisysRuntimeAPI",
    "llaisysStream_t",
    "llaisysTensor_t",
    "llaisysDataType_t",
    "DataType",
    "llaisysDeviceType_t",
    "DeviceType",
    "llaisysMemcpyKind_t",
    "MemcpyKind",
    "llaisysStream_t",
    "LlaisysQwen2Meta",
    "LlaisysQwen2Weights",
    "llaisysQwen2Model_t",
]
