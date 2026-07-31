"""Measure decode throughput as a function of context length.

The short-prompt benchmark hides attention cost: at kv=90 attention is ~13% of a
token, so a faster attention kernel barely moves the total. Attention is the one
part of decode that grows with context, so this sweeps the starting context to
show where kernel work actually matters.

Cost per step is recovered by differencing two runs of different length rather
than by subtracting a separate prefill measurement. Subtracting prefill fails at
long context: at ctx=16384 prefill is ~30 s while 31 decode steps are ~0.3 s, so
the decode signal is 1% of the subtrahend and prefill jitter alone swamps it --
the difference goes negative and a `max(..., 0)` clamp turns that into an
absurdly fast reading. Differencing two generate() calls that both pay the same
prefill cancels it exactly instead.

    python test/benchmark_context.py --model /root/models/... --device nvidia \
        --contexts 128,512,1024,2048 --steps 32 --tag optimized --out ctx.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import llaisys
from llaisys import DeviceType

DEVICES = {"cpu": DeviceType.CPU, "nvidia": DeviceType.NVIDIA, "metax": DeviceType.METAX}

# A difference smaller than this multiple of the observed run-to-run spread is
# not distinguishable from noise, so it is reported as an error instead of a
# number that looks like a measurement.
MIN_SIGNAL_TO_NOISE = 3.0


class UnreliableMeasurement(RuntimeError):
    """Raised when the timing difference is not separable from run-to-run noise."""


def _time_generate(model, prompt, new_tokens):
    """Wall seconds for one generate() call, plus how many tokens it really made.

    The token count is returned so truncation (an EOS hit inside the loop) shows
    up as a mismatch rather than as a fast-looking per-token average over steps
    that were never executed.
    """
    start = time.perf_counter()
    output = model.generate(
        prompt, max_new_tokens=new_tokens, top_k=1, top_p=1.0, temperature=0.0
    )
    elapsed = time.perf_counter() - start
    return elapsed, len(output) - len(prompt)


def _sample(model, prompt, new_tokens, repeats):
    """Median seconds over `repeats` timed calls, after one untimed warmup."""
    _time_generate(model, prompt, new_tokens)  # warmup: allocations, autotune, clocks

    samples = []
    for _ in range(repeats):
        elapsed, produced = _time_generate(model, prompt, new_tokens)
        if produced != new_tokens:
            raise UnreliableMeasurement(
                f"asked for {new_tokens} new tokens but generation stopped after "
                f"{produced} (EOS or cache limit); per-token timing would average "
                "over steps that never ran"
            )
        samples.append(elapsed)
    return statistics.median(samples), max(samples) - min(samples)


def decode_throughput(model, context_tokens, steps, repeats):
    """Milliseconds per decode step at a given starting context length.

    Times generation of `steps` and of `2 * steps` new tokens and divides the
    difference by the extra steps. Both runs prefill the same prompt, so the
    prefill term cancels algebraically and never enters the quotient.
    """
    prompt = [151646] + [100] * (context_tokens - 1)
    lo_steps, hi_steps = steps, steps * 2

    lo_time, lo_spread = _sample(model, prompt, lo_steps, repeats)
    hi_time, hi_spread = _sample(model, prompt, hi_steps, repeats)

    delta = hi_time - lo_time
    extra_steps = hi_steps - lo_steps
    # Worst-case error on `delta` if both medians sat at opposite ends of their
    # observed spread. Compared against the signal to catch cancellation.
    noise = lo_spread + hi_spread
    if delta <= 0 or delta < MIN_SIGNAL_TO_NOISE * noise:
        raise UnreliableMeasurement(
            f"timing difference {delta * 1000:.2f} ms over {extra_steps} steps is "
            f"within noise (spread {noise * 1000:.2f} ms); raise --steps or --repeats"
        )

    per_step = delta / extra_steps
    return {
        "context_tokens": context_tokens,
        "steps_low": lo_steps,
        "steps_high": hi_steps,
        "seconds_low": lo_time,
        "seconds_high": hi_time,
        "ms_per_token": per_step * 1000,
        "ms_per_token_uncertainty": noise / extra_steps * 1000,
        "tokens_per_second": 1.0 / per_step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="nvidia", choices=sorted(DEVICES))
    parser.add_argument("--contexts", default="128,512,1024,2048,4096")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    contexts = [int(value) for value in args.contexts.split(",")]
    model = llaisys.models.Qwen2(args.model, DEVICES[args.device])

    print(f"tag={args.tag} device={args.device} steps={args.steps} repeats={args.repeats}\n")
    print(f"{'context':>9}{'ms/token':>11}{'+/-':>9}{'tok/s':>10}")

    results = []
    failures = []
    for context in contexts:
        try:
            entry = decode_throughput(model, context, args.steps, args.repeats)
        except UnreliableMeasurement as error:
            failures.append((context, str(error)))
            print(f"{context:>9}{'unreliable':>11}")
            continue
        results.append(entry)
        print(
            f"{context:>9}{entry['ms_per_token']:>11.3f}"
            f"{entry['ms_per_token_uncertainty']:>9.3f}{entry['tokens_per_second']:>10.1f}"
        )

    if args.out:
        payload = {"tag": args.tag, "device": args.device, "steps": args.steps,
                   "results": results}
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    if failures:
        print()
        for context, message in failures:
            print(f"context={context}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
