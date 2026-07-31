"""Turns backend failures into Python exceptions.

The C API never throws: `src/llaisys/error.hpp` catches everything at the
boundary and records the message for the calling thread. Every exported function
is therefore only reported as failed through `llaisysGetLastError()`, so the
ctypes prototypes install an `errcheck` that polls it and raises.
"""

from ctypes import c_char_p


class LlaisysError(RuntimeError):
    """Raised when a llaisys backend call reports an error."""


def load_error(lib):
    lib.llaisysGetLastError.argtypes = []
    lib.llaisysGetLastError.restype = c_char_p

    lib.llaisysClearLastError.argtypes = []
    lib.llaisysClearLastError.restype = None


def check_error(lib, name):
    """Build an ``errcheck`` that raises whatever the backend recorded."""

    def errcheck(result, func, args):
        message = lib.llaisysGetLastError()
        if message is not None:
            lib.llaisysClearLastError()
            raise LlaisysError(f"{name}: {message.decode('utf-8', 'replace')}")
        return result

    return errcheck


# Reported through llaisysGetLastError() rather than raising, so they must not
# be wrapped (and wrapping them would recurse).
_UNCHECKED = frozenset({"llaisysGetLastError", "llaisysClearLastError"})


def install_error_checks(lib, names):
    """Attach the raising ``errcheck`` to each named export."""
    for name in names:
        if name in _UNCHECKED:
            continue
        prototype = getattr(lib, name)
        prototype.errcheck = check_error(lib, name)
