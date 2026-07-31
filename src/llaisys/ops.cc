#include "llaisys/ops.h"

#include "error.hpp"
#include "llaisys_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

using llaisys::capi::protect;

__C {
    void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b) {
        protect([&] { llaisys::ops::add(c->tensor, a->tensor, b->tensor); });
    }
    void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals) {
        protect([&] { llaisys::ops::argmax(max_idx->tensor, max_val->tensor, vals->tensor); });
    }
    void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight) {
        protect([&] { llaisys::ops::embedding(out->tensor, index->tensor, weight->tensor); });
    }
    void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias) {
        protect([&] {
            llaisys::ops::linear(out->tensor, in->tensor, weight->tensor,
                                 bias == nullptr ? nullptr : bias->tensor);
        });
    }
    void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in) {
        protect([&] { llaisys::ops::rearrange(out->tensor, in->tensor); });
    }
    void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps) {
        protect([&] { llaisys::ops::rms_norm(out->tensor, in->tensor, weight->tensor, eps); });
    }
    void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta) {
        protect([&] { llaisys::ops::rope(out->tensor, in->tensor, pos_ids->tensor, theta); });
    }
    void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale) {
        protect([&] {
            llaisys::ops::self_attention(attn_val->tensor, q->tensor, k->tensor, v->tensor, scale);
        });
    }
    void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up) {
        protect([&] { llaisys::ops::swiglu(out->tensor, gate->tensor, up->tensor); });
    }
}
