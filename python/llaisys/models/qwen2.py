from ctypes import c_int, c_int64, c_void_p
import json
import re
from typing import Sequence

from ..libllaisys import DataType, DeviceType, LIB_LLAISYS, LlaisysQwen2Meta

from pathlib import Path
import safetensors


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        with (model_path / "config.json").open(encoding="utf-8") as file:
            config = json.load(file)

        dtype_by_name = {
            "float16": DataType.F16,
            "float32": DataType.F32,
            "bfloat16": DataType.BF16,
        }
        dtype = dtype_by_name[config["torch_dtype"]]
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        head_dim = config.get("head_dim", hidden_size // num_heads)
        self._meta = LlaisysQwen2Meta(
            dtype,
            config["num_hidden_layers"],
            hidden_size,
            num_heads,
            config["num_key_value_heads"],
            head_dim,
            config["intermediate_size"],
            config["max_position_embeddings"],
            config["vocab_size"],
            config["rms_norm_eps"],
            config["rope_theta"],
            config["eos_token_id"],
        )
        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            self._meta, device, device_ids, 1
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents
        layer_fields = {
            "input_layernorm.weight": "attn_norm_w",
            "self_attn.q_proj.weight": "attn_q_w",
            "self_attn.q_proj.bias": "attn_q_b",
            "self_attn.k_proj.weight": "attn_k_w",
            "self_attn.k_proj.bias": "attn_k_b",
            "self_attn.v_proj.weight": "attn_v_w",
            "self_attn.v_proj.bias": "attn_v_b",
            "self_attn.o_proj.weight": "attn_o_w",
            "post_attention_layernorm.weight": "mlp_norm_w",
            "mlp.gate_proj.weight": "mlp_gate_w",
            "mlp.up_proj.weight": "mlp_up_w",
            "mlp.down_proj.weight": "mlp_down_w",
        }
        fixed_fields = {
            "model.embed_tokens.weight": weights.in_embed,
            "lm_head.weight": weights.out_embed,
            "model.norm.weight": weights.out_norm_w,
        }
        loaded = set()

        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                target = fixed_fields.get(name_)
                if target is None:
                    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name_)
                    if match is None or match.group(2) not in layer_fields:
                        raise ValueError(f"Unexpected Qwen2 weight: {name_}")
                    layer = int(match.group(1))
                    if layer >= self._meta.nlayer:
                        raise ValueError(f"Qwen2 layer index out of range: {layer}")
                    target = getattr(weights, layer_fields[match.group(2)])[layer]

                value = data_.get_tensor(name_)
                LIB_LLAISYS.tensorLoad(target, c_void_p(value.data_ptr()))
                loaded.add(name_)

        expected_weights = 3 + 12 * self._meta.nlayer
        if len(loaded) != expected_weights:
            raise ValueError(
                f"Incomplete Qwen2 checkpoint: loaded {len(loaded)} of {expected_weights} weights"
            )

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 0.0,
    ):
        """Greedy-decode `inputs` and return prompt + generated tokens.

        Sampling is not implemented in the backend yet: the only accepted values
        are the ones that reduce to greedy decoding. Anything else raises rather
        than silently downgrading, so callers cannot believe they got sampling.
        """
        if not inputs:
            raise ValueError("Qwen2 generation requires at least one input token")
        if top_k != 1:
            raise ValueError(
                f"Qwen2 backend supports greedy decoding only; got top_k={top_k} (expected 1)"
            )
        if temperature not in (0.0, 1.0):
            raise ValueError(
                f"Qwen2 backend supports greedy decoding only; got temperature={temperature} "
                "(expected 0.0)"
            )
        if top_p != 1.0:
            raise ValueError(
                f"Qwen2 backend supports greedy decoding only; got top_p={top_p} (expected 1.0)"
            )

        max_new_tokens = 128 if max_new_tokens is None else max_new_tokens
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        output = list(inputs)
        pending = list(inputs)
        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)
        cache_capacity = len(inputs) + max_new_tokens
        if cache_capacity > self._meta.maxseq:
            raise ValueError(
                f"Requested sequence length {cache_capacity} exceeds model limit {self._meta.maxseq}"
            )
        LIB_LLAISYS.llaisysQwen2ModelReserveCache(self._model, max(1, cache_capacity))
        for _ in range(max_new_tokens):
            token_buffer = (c_int64 * len(pending))(*pending)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model, token_buffer, len(pending)
                )
            )
            output.append(next_token)
            if next_token == self._meta.end_token:
                break
            pending = [next_token]

        return output
