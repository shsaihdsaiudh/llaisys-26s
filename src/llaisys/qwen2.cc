#include "llaisys/models/qwen2.h"

#include "error.hpp"
#include "llaisys_tensor.hpp"

#include "../tensor/tensor.hpp"
#include "../utils.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

struct Qwen2Workspace {
    size_t capacity = 0;
    llaisys::tensor_t token_ids;
    llaisys::tensor_t positions;
    llaisys::tensor_t hidden;
    llaisys::tensor_t normalized;
    llaisys::tensor_t query;
    llaisys::tensor_t attention;
    llaisys::tensor_t projected;
    llaisys::tensor_t gate;
    llaisys::tensor_t up;
    llaisys::tensor_t down;
    llaisys::tensor_t logits;
    llaisys::tensor_t max_index;
    llaisys::tensor_t max_value;
    std::vector<int64_t> position_values;
};

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;
    int device_id;
    LlaisysQwen2Weights weights{};
    std::vector<llaisys::tensor_t> key_cache;
    std::vector<llaisys::tensor_t> value_cache;
    size_t cache_length = 0;
    size_t cache_capacity = 0;
    Qwen2Workspace workspace;
};

namespace {
llaisysTensor_t make_tensor(const std::vector<size_t> &shape,
                            llaisysDataType_t dtype,
                            llaisysDeviceType_t device,
                            int device_id) {
    return new LlaisysTensor{llaisys::Tensor::create(shape, dtype, device, device_id)};
}

void destroy_tensor(llaisysTensor_t tensor) {
    delete tensor;
}

llaisysTensor_t *make_layer_array(size_t layers) {
    return new llaisysTensor_t[layers]{};
}

void destroy_layer_array(llaisysTensor_t *tensors, size_t layers) {
    if (tensors == nullptr) {
        return;
    }
    for (size_t layer = 0; layer < layers; ++layer) {
        destroy_tensor(tensors[layer]);
    }
    delete[] tensors;
}

llaisys::tensor_t tensor(llaisysTensor_t handle) {
    return handle->tensor;
}

Qwen2Workspace make_workspace(const LlaisysQwen2Model &model, size_t capacity) {
    const auto &meta = model.meta;
    Qwen2Workspace workspace;
    workspace.capacity = capacity;
    workspace.token_ids = llaisys::Tensor::create(
        {capacity}, LLAISYS_DTYPE_I64, model.device, model.device_id);
    workspace.positions = llaisys::Tensor::create(
        {capacity}, LLAISYS_DTYPE_I64, model.device, model.device_id);
    workspace.hidden = llaisys::Tensor::create(
        {capacity, meta.hs}, meta.dtype, model.device, model.device_id);
    workspace.normalized = llaisys::Tensor::create(
        {capacity, meta.hs}, meta.dtype, model.device, model.device_id);
    workspace.query = llaisys::Tensor::create(
        {capacity, meta.nh * meta.dh}, meta.dtype, model.device, model.device_id);
    workspace.attention = llaisys::Tensor::create(
        {capacity, meta.nh, meta.dh}, meta.dtype, model.device, model.device_id);
    workspace.projected = llaisys::Tensor::create(
        {capacity, meta.hs}, meta.dtype, model.device, model.device_id);
    workspace.gate = llaisys::Tensor::create(
        {capacity, meta.di}, meta.dtype, model.device, model.device_id);
    workspace.up = llaisys::Tensor::create(
        {capacity, meta.di}, meta.dtype, model.device, model.device_id);
    workspace.down = llaisys::Tensor::create(
        {capacity, meta.hs}, meta.dtype, model.device, model.device_id);
    workspace.logits = llaisys::Tensor::create(
        {1, meta.voc}, meta.dtype, model.device, model.device_id);
    workspace.max_index = llaisys::Tensor::create(
        {1}, LLAISYS_DTYPE_I64, model.device, model.device_id);
    workspace.max_value = llaisys::Tensor::create(
        {1}, meta.dtype, model.device, model.device_id);
    workspace.position_values.resize(capacity);
    return workspace;
}

void ensure_workspace(LlaisysQwen2Model *model, size_t capacity) {
    // The workspace only ever grows. Decoding alternates between a long prefill
    // (ntoken == prompt length) and single-token steps, so reallocating on every
    // size change would rebuild every buffer twice per request.
    if (capacity <= model->workspace.capacity) {
        return;
    }
    llaisys::core::context().setDevice(model->device, model->device_id);
    model->workspace = {};
    model->workspace = make_workspace(*model, capacity);
}

LlaisysQwen2Model *modelCreate(const LlaisysQwen2Meta *meta,
                               llaisysDeviceType_t device,
                               int *device_ids,
                               int ndevice) {
    CHECK_ARGUMENT(meta != nullptr, "Qwen2 metadata must not be null");
    bool supported_device = device == LLAISYS_DEVICE_CPU;
#ifdef ENABLE_NVIDIA_API
    supported_device = supported_device || device == LLAISYS_DEVICE_NVIDIA;
#endif
#ifdef ENABLE_METAX_API
    supported_device = supported_device || device == LLAISYS_DEVICE_METAX;
#endif
    CHECK_ARGUMENT(supported_device, "Qwen2 device backend is not enabled in this build");
    CHECK_ARGUMENT(ndevice == 1 && device_ids != nullptr, "Qwen2 requires exactly one device");
    CHECK_ARGUMENT(meta->nlayer > 0 && meta->hs > 0 && meta->nh > 0 && meta->nkvh > 0,
                   "Qwen2 dimensions must be positive");
    CHECK_ARGUMENT(meta->hs == meta->nh * meta->dh, "Qwen2 hidden size must equal heads times head dimension");
    CHECK_ARGUMENT(meta->nh % meta->nkvh == 0, "Qwen2 query heads must be divisible by KV heads");
    CHECK_ARGUMENT(meta->di > 0 && meta->maxseq > 0 && meta->voc > 0,
                   "Qwen2 intermediate, sequence, and vocabulary sizes must be positive");

    auto *model = new LlaisysQwen2Model{*meta, device, device_ids[0]};
    auto &weights = model->weights;

    weights.in_embed = make_tensor({meta->voc, meta->hs}, meta->dtype, device, model->device_id);
    weights.out_embed = make_tensor({meta->voc, meta->hs}, meta->dtype, device, model->device_id);
    weights.out_norm_w = make_tensor({meta->hs}, meta->dtype, device, model->device_id);

    weights.attn_norm_w = make_layer_array(meta->nlayer);
    weights.attn_q_w = make_layer_array(meta->nlayer);
    weights.attn_q_b = make_layer_array(meta->nlayer);
    weights.attn_k_w = make_layer_array(meta->nlayer);
    weights.attn_k_b = make_layer_array(meta->nlayer);
    weights.attn_v_w = make_layer_array(meta->nlayer);
    weights.attn_v_b = make_layer_array(meta->nlayer);
    weights.attn_o_w = make_layer_array(meta->nlayer);
    weights.mlp_norm_w = make_layer_array(meta->nlayer);
    weights.mlp_gate_w = make_layer_array(meta->nlayer);
    weights.mlp_up_w = make_layer_array(meta->nlayer);
    weights.mlp_down_w = make_layer_array(meta->nlayer);

    for (size_t layer = 0; layer < meta->nlayer; ++layer) {
        weights.attn_norm_w[layer] = make_tensor({meta->hs}, meta->dtype, device, model->device_id);
        weights.attn_q_w[layer] = make_tensor({meta->nh * meta->dh, meta->hs}, meta->dtype, device, model->device_id);
        weights.attn_q_b[layer] = make_tensor({meta->nh * meta->dh}, meta->dtype, device, model->device_id);
        weights.attn_k_w[layer] = make_tensor({meta->nkvh * meta->dh, meta->hs}, meta->dtype, device, model->device_id);
        weights.attn_k_b[layer] = make_tensor({meta->nkvh * meta->dh}, meta->dtype, device, model->device_id);
        weights.attn_v_w[layer] = make_tensor({meta->nkvh * meta->dh, meta->hs}, meta->dtype, device, model->device_id);
        weights.attn_v_b[layer] = make_tensor({meta->nkvh * meta->dh}, meta->dtype, device, model->device_id);
        weights.attn_o_w[layer] = make_tensor({meta->hs, meta->hs}, meta->dtype, device, model->device_id);
        weights.mlp_norm_w[layer] = make_tensor({meta->hs}, meta->dtype, device, model->device_id);
        weights.mlp_gate_w[layer] = make_tensor({meta->di, meta->hs}, meta->dtype, device, model->device_id);
        weights.mlp_up_w[layer] = make_tensor({meta->di, meta->hs}, meta->dtype, device, model->device_id);
        weights.mlp_down_w[layer] = make_tensor({meta->hs, meta->di}, meta->dtype, device, model->device_id);
    }

    return model;
}

void modelDestroy(LlaisysQwen2Model *model) {
    if (model == nullptr) {
        return;
    }
    auto &weights = model->weights;
    destroy_tensor(weights.in_embed);
    destroy_tensor(weights.out_embed);
    destroy_tensor(weights.out_norm_w);
    destroy_layer_array(weights.attn_norm_w, model->meta.nlayer);
    destroy_layer_array(weights.attn_q_w, model->meta.nlayer);
    destroy_layer_array(weights.attn_q_b, model->meta.nlayer);
    destroy_layer_array(weights.attn_k_w, model->meta.nlayer);
    destroy_layer_array(weights.attn_k_b, model->meta.nlayer);
    destroy_layer_array(weights.attn_v_w, model->meta.nlayer);
    destroy_layer_array(weights.attn_v_b, model->meta.nlayer);
    destroy_layer_array(weights.attn_o_w, model->meta.nlayer);
    destroy_layer_array(weights.mlp_norm_w, model->meta.nlayer);
    destroy_layer_array(weights.mlp_gate_w, model->meta.nlayer);
    destroy_layer_array(weights.mlp_up_w, model->meta.nlayer);
    destroy_layer_array(weights.mlp_down_w, model->meta.nlayer);
    delete model;
}

void modelReserveCache(LlaisysQwen2Model *model, size_t capacity) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
    CHECK_ARGUMENT(capacity > 0 && capacity <= model->meta.maxseq,
                   "Qwen2 cache capacity is outside the configured sequence range");
    if (capacity == model->cache_capacity
        && model->key_cache.size() == model->meta.nlayer
        && model->value_cache.size() == model->meta.nlayer) {
        return;
    }
    CHECK_ARGUMENT(model->cache_length == 0, "Qwen2 cache can only be resized after reset");

    llaisys::core::context().setDevice(model->device, model->device_id);
    model->key_cache.clear();
    model->value_cache.clear();
    model->cache_capacity = 0;

    std::vector<llaisys::tensor_t> key_cache;
    std::vector<llaisys::tensor_t> value_cache;
    key_cache.reserve(model->meta.nlayer);
    value_cache.reserve(model->meta.nlayer);
    for (size_t layer = 0; layer < model->meta.nlayer; ++layer) {
        key_cache.push_back(llaisys::Tensor::create(
            {capacity, model->meta.nkvh, model->meta.dh}, model->meta.dtype,
            model->device, model->device_id));
        value_cache.push_back(llaisys::Tensor::create(
            {capacity, model->meta.nkvh, model->meta.dh}, model->meta.dtype,
            model->device, model->device_id));
    }
    model->key_cache.swap(key_cache);
    model->value_cache.swap(value_cache);
    model->cache_capacity = capacity;
}

int64_t modelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
    CHECK_ARGUMENT(token_ids != nullptr && ntoken > 0, "Qwen2 inference requires at least one token");
    CHECK_ARGUMENT(model->cache_capacity > 0 && model->key_cache.size() == model->meta.nlayer,
                   "Qwen2 cache must be reserved before inference");
    CHECK_ARGUMENT(model->cache_length + ntoken <= model->cache_capacity,
                   "Qwen2 sequence exceeds the reserved cache capacity");
    for (size_t i = 0; i < ntoken; ++i) {
        CHECK_ARGUMENT(token_ids[i] >= 0 && static_cast<size_t>(token_ids[i]) < model->meta.voc,
                       "Qwen2 token ID is out of range");
    }

    const auto &meta = model->meta;
    auto &weights = model->weights;
    ensure_workspace(model, ntoken);
    auto &workspace = model->workspace;
    auto tokens = workspace.token_ids->slice(0, 0, ntoken);
    auto positions = workspace.positions->slice(0, 0, ntoken);
    auto hidden = workspace.hidden->slice(0, 0, ntoken);
    auto normalized = workspace.normalized->slice(0, 0, ntoken);
    auto query_2d = workspace.query->slice(0, 0, ntoken);
    auto attention = workspace.attention->slice(0, 0, ntoken);
    auto projected = workspace.projected->slice(0, 0, ntoken);
    auto gate = workspace.gate->slice(0, 0, ntoken);
    auto up = workspace.up->slice(0, 0, ntoken);
    auto down = workspace.down->slice(0, 0, ntoken);

    tokens->load(token_ids);
    llaisys::ops::embedding(hidden, tokens, tensor(weights.in_embed));

    for (size_t i = 0; i < ntoken; ++i) {
        workspace.position_values[i] = static_cast<int64_t>(model->cache_length + i);
    }
    positions->load(workspace.position_values.data());

    const size_t total_length = model->cache_length + ntoken;
    const size_t kv_width = meta.nkvh * meta.dh;

    for (size_t layer = 0; layer < meta.nlayer; ++layer) {
        llaisys::ops::rms_norm(normalized, hidden, tensor(weights.attn_norm_w[layer]), meta.epsilon);

        auto cached_keys = model->key_cache[layer]->slice(0, 0, total_length);
        auto cached_values = model->value_cache[layer]->slice(0, 0, total_length);
        auto key = cached_keys->slice(0, model->cache_length, total_length);
        auto value = cached_values->slice(0, model->cache_length, total_length);
        llaisys::ops::linear(query_2d, normalized,
                             tensor(weights.attn_q_w[layer]), tensor(weights.attn_q_b[layer]));
        llaisys::ops::linear(key->view({ntoken, kv_width}), normalized,
                             tensor(weights.attn_k_w[layer]), tensor(weights.attn_k_b[layer]));
        llaisys::ops::linear(value->view({ntoken, kv_width}), normalized,
                             tensor(weights.attn_v_w[layer]), tensor(weights.attn_v_b[layer]));

        auto query = query_2d->view({ntoken, meta.nh, meta.dh});
        llaisys::ops::rope(query, query, positions, meta.theta);
        llaisys::ops::rope(key, key, positions, meta.theta);

        llaisys::ops::self_attention(
            attention, query, cached_keys, cached_values, 1.0F / std::sqrt(static_cast<float>(meta.dh)));
        llaisys::ops::linear(
            projected, attention->view({ntoken, meta.hs}), tensor(weights.attn_o_w[layer]), nullptr);
        llaisys::ops::add(hidden, hidden, projected);

        llaisys::ops::rms_norm(normalized, hidden, tensor(weights.mlp_norm_w[layer]), meta.epsilon);
        llaisys::ops::linear(gate, normalized, tensor(weights.mlp_gate_w[layer]), nullptr);
        llaisys::ops::linear(up, normalized, tensor(weights.mlp_up_w[layer]), nullptr);
        llaisys::ops::swiglu(gate, gate, up);
        llaisys::ops::linear(down, gate, tensor(weights.mlp_down_w[layer]), nullptr);
        llaisys::ops::add(hidden, hidden, down);
    }

    model->cache_length = total_length;
    auto last_hidden = hidden->slice(0, ntoken - 1, ntoken);
    auto output_normalized = normalized->slice(0, 0, 1);
    llaisys::ops::rms_norm(output_normalized, last_hidden, tensor(weights.out_norm_w), meta.epsilon);
    auto logits = workspace.logits;
    llaisys::ops::linear(logits, output_normalized, tensor(weights.out_embed), nullptr);
    auto max_index = workspace.max_index;
    auto max_value = workspace.max_value;
    llaisys::ops::argmax(max_index, max_value, logits->view({meta.voc}));
    int64_t result = 0;
    if (model->device == LLAISYS_DEVICE_CPU) {
        result = *reinterpret_cast<const int64_t *>(max_index->data());
    } else {
        llaisys::core::context().setDevice(model->device, model->device_id);
        llaisys::core::context().runtime().api()->memcpy_sync(
            &result, max_index->data(), sizeof(result), LLAISYS_MEMCPY_D2H);
    }
    return result;
}
} // namespace

using llaisys::capi::protect;

__C {
    LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                               llaisysDeviceType_t device,
                                               int *device_ids,
                                               int ndevice) {
        return protect([&] { return modelCreate(meta, device, device_ids, ndevice); });
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model * model) {
        protect([&] { modelDestroy(model); });
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model * model) {
        return protect([&] {
            CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
            return &model->weights;
        });
    }

    void llaisysQwen2ModelReset(LlaisysQwen2Model * model) {
        protect([&] {
            CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
            model->cache_length = 0;
        });
    }

    void llaisysQwen2ModelReserveCache(LlaisysQwen2Model * model, size_t capacity) {
        protect([&] { modelReserveCache(model, capacity); });
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
        return protect([&] { return modelInfer(model, token_ids, ntoken); });
    }
}
