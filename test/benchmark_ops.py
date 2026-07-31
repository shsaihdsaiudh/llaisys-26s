"""Time individual operators at Qwen2 decode geometry.

`ncu` needs GPU performance counters, which containers usually withhold, so this
attributes decode cost by timing each op in isolation instead. Numbers are
per-call microseconds; multiply by calls-per-token (printed) to see each op's
share of a decode step.

    python test/benchmark_ops.py --device nvidia --kv 128
"""

import argparse
import statistics
import time

from test_utils import llaisys_device

import llaisys
from llaisys import DataType, DeviceType

# DeepSeek-R1-Distill-Qwen-1.5B geometry.
LAYERS = 28
HIDDEN = 1536
HEADS = 12
KV_HEADS = 2
HEAD_DIM = 128
INTERMEDIATE = 8960
VOCAB = 151936


def make(shape, dtype, device):
    return llaisys.Tensor(shape=shape, dtype=dtype, device=device)


def timed(label, fn, calls_per_token, runtime, repeats=50, warmup=10):
    for _ in range(warmup):
        fn()
    runtime.device_synchronize()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        runtime.device_synchronize()
        samples.append((time.perf_counter() - start) * 1e6)

    median = statistics.median(samples)
    return {
        "op": label,
        "microseconds": median,
        "calls_per_token": calls_per_token,
        "token_microseconds": median * calls_per_token,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia", "metax"])
    parser.add_argument("--kv", type=int, default=128, help="KV cache length to simulate")
    parser.add_argument("--dtype", default="bf16", choices=["f32", "f16", "bf16"])
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    dtype = {"f32": DataType.F32, "f16": DataType.F16, "bf16": DataType.BF16}[args.dtype]
    device = llaisys_device(args.device)
    llaisys.set_context_runtime(device, 0)
    runtime = llaisys.RuntimeAPI(device)

    kv = args.kv
    rows = 1  # single-token decode

    hidden = make((rows, HIDDEN), dtype, device)
    normalized = make((rows, HIDDEN), dtype, device)
    norm_w = make((HIDDEN,), dtype, device)
    q = make((rows, HEADS * HEAD_DIM), dtype, device)
    q3 = q.view(rows, HEADS, HEAD_DIM)
    qw = make((HEADS * HEAD_DIM, HIDDEN), dtype, device)
    qb = make((HEADS * HEAD_DIM,), dtype, device)
    keys = make((kv, KV_HEADS, HEAD_DIM), dtype, device)
    values = make((kv, KV_HEADS, HEAD_DIM), dtype, device)
    attn = make((rows, HEADS, HEAD_DIM), dtype, device)
    ow = make((HIDDEN, HIDDEN), dtype, device)
    projected = make((rows, HIDDEN), dtype, device)
    gate = make((rows, INTERMEDIATE), dtype, device)
    up = make((rows, INTERMEDIATE), dtype, device)
    gate_w = make((INTERMEDIATE, HIDDEN), dtype, device)
    down = make((rows, HIDDEN), dtype, device)
    down_w = make((HIDDEN, INTERMEDIATE), dtype, device)
    positions = make((rows,), DataType.I64, device)
    tokens = make((rows,), DataType.I64, device)
    embed_w = make((VOCAB, HIDDEN), dtype, device)
    logits = make((rows, VOCAB), dtype, device)
    logits_1d = logits.view(VOCAB)
    max_index = make((1,), DataType.I64, device)
    max_value = make((1,), dtype, device)

    print(f"device={args.device} dtype={args.dtype} kv_len={kv} layers={LAYERS}\n")

    results = [
        timed("embedding", lambda: llaisys.Ops.embedding(hidden, tokens, embed_w),
              1, runtime, args.repeats),
        timed("rms_norm", lambda: llaisys.Ops.rms_norm(normalized, hidden, norm_w, 1e-6),
              2 * LAYERS + 1, runtime, args.repeats),
        timed("linear_qkv_o", lambda: llaisys.Ops.linear(q, normalized, qw, qb),
              LAYERS, runtime, args.repeats),
        timed("linear_o", lambda: llaisys.Ops.linear(projected, hidden, ow, None),
              LAYERS, runtime, args.repeats),
        timed("linear_mlp_gate", lambda: llaisys.Ops.linear(gate, normalized, gate_w, None),
              2 * LAYERS, runtime, args.repeats),
        timed("linear_mlp_down", lambda: llaisys.Ops.linear(down, gate, down_w, None),
              LAYERS, runtime, args.repeats),
        timed("rope", lambda: llaisys.Ops.rope(q3, q3, positions, 10000.0),
              2 * LAYERS, runtime, args.repeats),
        timed("self_attention",
              lambda: llaisys.Ops.self_attention(attn, q3, keys, values, 0.088),
              LAYERS, runtime, args.repeats),
        timed("swiglu", lambda: llaisys.Ops.swiglu(gate, gate, up),
              LAYERS, runtime, args.repeats),
        timed("add", lambda: llaisys.Ops.add(hidden, hidden, projected),
              2 * LAYERS, runtime, args.repeats),
        timed("linear_lm_head", lambda: llaisys.Ops.linear(logits, normalized, embed_w, None),
              1, runtime, args.repeats),
        timed("argmax", lambda: llaisys.Ops.argmax(max_index, max_value, logits_1d),
              1, runtime, args.repeats),
    ]

    total = sum(item["token_microseconds"] for item in results)
    results.sort(key=lambda item: item["token_microseconds"], reverse=True)

    print(f"{'op':<18}{'us/call':>10}{'calls':>7}{'us/token':>11}{'share':>8}")
    for item in results:
        share = item["token_microseconds"] / total * 100 if total else 0
        print(f"{item['op']:<18}{item['microseconds']:>10.1f}{item['calls_per_token']:>7}"
              f"{item['token_microseconds']:>11.1f}{share:>7.1f}%")
    print(f"\n{'modelled total':<18}{'':>10}{'':>7}{total:>11.1f}{'':>8}")
    print(f"{'':<18}{'':>10}{'':>7}{total / 1000:>11.3f} ms/token")


if __name__ == "__main__":
    main()
