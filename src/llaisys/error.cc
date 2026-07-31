#include "error.hpp"

#include "llaisys/error.h"

namespace llaisys::capi {
namespace {
// Owned per thread so concurrent callers cannot clobber each other's message,
// and so the pointer handed to C stays alive until that thread's next call.
thread_local std::string message;
thread_local bool has_error = false;
} // namespace

void setLastError(std::string value) {
    message = std::move(value);
    has_error = true;
}

const char *lastError() {
    return has_error ? message.c_str() : nullptr;
}

void clearLastError() {
    has_error = false;
    message.clear();
}
} // namespace llaisys::capi

__C {
    const char *llaisysGetLastError() {
        return llaisys::capi::lastError();
    }

    void llaisysClearLastError() {
        llaisys::capi::clearLastError();
    }
}
