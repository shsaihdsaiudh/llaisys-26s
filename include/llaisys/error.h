#ifndef LLAISYS_ERROR_H
#define LLAISYS_ERROR_H

#include "../llaisys.h"

/// Message describing why the most recent llaisys call on this thread failed,
/// or NULL if it succeeded. The pointer stays valid until the next llaisys call
/// on the same thread.
__C __export const char *llaisysGetLastError();

/// Discards the pending error for this thread.
__C __export void llaisysClearLastError();

#endif // LLAISYS_ERROR_H
