"""Benchmark LLAISYS Qwen2 decoding so optimizations can be compared.

Unlike test_infer.py (which measures a single wall-clock generate() next to a
Transformers run in the same process), this reports prefill and decode
separately, runs repeats, and emits JSON so before/after runs can be diffed.

    python test/benchmark_infer.py --model /root/models/... --device nvidia \
        --max_steps 128 --repeats 3 --tag baseline --out bench_baseline.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from test_utils import llaisys_device

import llaisys
from transformers import AutoTokenizer


def encode_prompt(tokenizer, prompt):
    content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(content)


def time_generate(model, tokens, max_new_tokens):
    """Return (total_seconds, generated_token_ids)."""
    start = time.perf_counter()
    out = model.generate(list(tokens), max_new_tokens=max_new_tokens, top_k=1,
                         top_p=1.0, temperature=0.0)
    return time.perf_counter() - start, out


def time_prefill(model, tokens):
    """Cost of the prompt pass alone: one generate() capped at a single step."""
    start = time.perf_counter()
    model.generate(list(tokens), max_new_tokens=1, top_k=1, top_p=1.0, temperature=0.0)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"])
    parser.add_argument("--prompt", default="Who are you?")
    parser.add_argument("--max_steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tag", default="run", help="label recorded in the JSON")
    parser.add_argument("--out", default=None, help="write JSON results here")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_tokens = encode_prompt(tokenizer, args.prompt)
    model = llaisys.models.Qwen2(args.model, llaisys_device(args.device))

    print(f"tag={args.tag} device={args.device} prompt_tokens={len(prompt_tokens)} "
          f"max_steps={args.max_steps}")

    for _ in range(args.warmup):
        time_generate(model, prompt_tokens, args.max_steps)

    totals, prefills, generated = [], [], None
    for index in range(args.repeats):
        elapsed, out = time_generate(model, prompt_tokens, args.max_steps)
        totals.append(elapsed)
        prefills.append(time_prefill(model, prompt_tokens))
        new_tokens = len(out) - len(prompt_tokens)
        generated = out
        print(f"  run {index + 1}: {elapsed:.4f}s total, {new_tokens} new tokens, "
              f"{new_tokens / elapsed:.1f} tok/s")

    new_tokens = len(generated) - len(prompt_tokens)
    total = statistics.median(totals)
    prefill = statistics.median(prefills)
    # Decode = everything after the prompt pass, so per-token cost excludes prefill.
    decode = max(total - prefill, 0.0)
    decode_steps = max(new_tokens - 1, 1)

    result = {
        "tag": args.tag,
        "device": args.device,
        "prompt": args.prompt,
        "prompt_tokens": len(prompt_tokens),
        "max_steps": args.max_steps,
        "repeats": args.repeats,
        "new_tokens": new_tokens,
        "total_seconds_median": total,
        "total_seconds_all": totals,
        "prefill_seconds_median": prefill,
        "decode_seconds_median": decode,
        "decode_tokens_per_second": decode_steps / decode if decode > 0 else None,
        "overall_tokens_per_second": new_tokens / total if total > 0 else None,
        "output_tokens": generated,
    }

    print(f"\nmedian total     {total:.4f}s")
    print(f"median prefill   {prefill:.4f}s ({len(prompt_tokens)} tokens)")
    print(f"median decode    {decode:.4f}s ({decode_steps} steps)")
    if result["decode_tokens_per_second"]:
        print(f"decode throughput {result['decode_tokens_per_second']:.1f} tok/s")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
