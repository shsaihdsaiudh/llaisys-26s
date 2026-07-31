#pragma once

#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace llaisys::capi {

// Records the message for the calling thread. Reported through
// llaisysGetLastError() so the C boundary never has to throw.
void setLastError(std::string message);

// The message recorded for the calling thread, or nullptr when the last call
// succeeded.
const char *lastError();

void clearLastError();

// Runs `fn` and converts any exception into a thread-local error message.
//
// Every function exported through `__C` must funnel through here: throwing
// across an `extern "C"` frame is undefined behaviour, and callers reach us
// through ctypes, which cannot unwind C++ frames. On failure a value-initialized
// result is returned (nullptr for handles, 0 for scalars) and the caller is
// expected to consult llaisysGetLastError().
template <typename Fn>
auto protect(Fn &&fn) -> decltype(fn()) {
    using Result = decltype(fn());
    clearLastError();
    try {
        if constexpr (std::is_void_v<Result>) {
            std::forward<Fn>(fn)();
            return;
        } else {
            return std::forward<Fn>(fn)();
        }
    } catch (const std::exception &error) {
        setLastError(error.what());
    } catch (...) {
        setLastError("unknown error");
    }
    if constexpr (!std::is_void_v<Result>) {
        return Result{};
    }
}

} // namespace llaisys::capi
