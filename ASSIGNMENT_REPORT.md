# LLAISYS Assignment Reproduction Report

## Platform status

| Platform | Runtime | Operators | Qwen2 inference | Status |
| --- | --- | --- | --- | --- |
| CPU (Linux, GCC) | Passed | Passed | Passed | Supported |
| CPU (Windows, MSVC 2022) | Passed | Passed | Not run locally (CI) | Supported |
| NVIDIA RTX 4090 | Passed | Passed | Passed | Supported |
| MetaX C500 | Passed | Passed | Passed | Supported |

Operator coverage includes Float32, Float16, and BFloat16 on every platform. The
Qwen2 pipeline (loader, KV-cache decode, and the 128-step exact-match generation
test) passes on CPU, NVIDIA, and MetaX.

Windows is listed separately because it is a distinct compiler, not just a
distinct OS: MSVC implements OpenMP 2.0, which rejects `collapse` and requires a
signed loop index, so all five OpenMP CPU kernels failed to compile there while
building cleanly under GCC. CI builds on `windows-latest`, so this was a real
break rather than a portability nicety. It was found by reproducing the CI build
locally with MSVC 2022 and is fixed; runtime, tensor, and all eight operator
suites now pass on Windows/CPU.

## Architecture

LLAISYS is a homogeneous-hardware framework: each thread owns a thread-local
`Context` that lazily creates one `Runtime` per device and switches the active
device on demand. Storage keeps the runtime that allocated it, so a tensor can
safely outlive the thread that created it.

Both GPU backends share a single, vendor-neutral kernel layer. Each operator has
one CUDA-compatible implementation under `src/ops/*/cuda/`; the NVIDIA (`.cu`)
and MetaX (`.maca`) entry files are one-line includes compiled by NVCC and MXCC
respectively. Neither vendor backend depends on the other, and every bug fix or
optimization lands in one place.

```
                 include/llaisys/*.h   (C API, __export)
                          |
        python/llaisys  --+--  src/llaisys/*.cc   (C API impl, e.g. qwen2.cc)
        (ctypes wrappers) |
                          v
              src/ops/<op>/op.cpp   (validation + CPU kernel + dispatch)
                          |
             +------------+-------------------------------+
             |                                            |
        CPU (OpenMP)                        src/ops/<op>/cuda/<op>_cuda.cuh
                                            (single vendor-neutral kernel)
                                                |                    |
                              src/ops/<op>/nvidia/*.cu   src/ops/<op>/metax/*.maca
                                    (NVCC, 1-line)            (MXCC, 1-line)
                                                |                    |
                              src/device/nvidia/*         src/device/metax/*
                              (CUDA Runtime API)          (MACA Runtime API)
```

### Design highlights

- **Single-source kernels.** Eight operators, two GPU vendors, zero duplicated
  kernel code. Adding a third CUDA-like vendor means adding thin entry files and
  a runtime-API implementation — no kernel rewrites.
- **Mixed-precision correctness.** Every Float16/BFloat16 kernel accumulates in
  Float32 (`toFloat`/`fromFloat` helpers in `src/ops/cuda/common.cuh`), so
  reductions such as RMS-norm and attention softmax keep full-precision math.
- **Fused single-token attention.** Decoding a single token runs a fused
  attention kernel with an in-block softmax — no global score buffer and no
  per-layer `cudaMalloc`/`cudaFree`.
- **Request-sized KV cache and reusable workspace.** The cache is reserved from
  `input_tokens + max_new_tokens` rather than the model's 131,072-token maximum,
  and inference reuses a model-owned workspace instead of allocating hundreds of
  temporary tensors per generated token.
- **RAII lifetime management.** The thread-local context owns runtimes with RAII;
  storage retains its allocating runtime; lazy device switching stores newly
  created runtimes in the context instead of leaking a copied entry.

## NVIDIA environment

- GPU: NVIDIA GeForce RTX 4090, 24 GB, compute capability 8.9
- Driver: 570.124.06
- OS: Ubuntu 24.04.1 LTS
- CUDA Toolkit: 12.8
- Xmake: 3.0.9
- Python: 3.12.3
- PyTorch: 2.6.0 NGC build

## MetaX environment

Two C500 hosts were used. The optimization measurements and the TF32 finding come
from the second, which has a larger quota:

- GPU: MetaX C500 — host A: 16 GB sGPU quota, 25% compute; host B: 32 GB sGPU
  quota, 50% compute, 128 cores
- Driver: 3.8.30
- MACA: 3.3.0.15
- MXCC: 1.0.0
- OS: Ubuntu 24.04.1 LTS
- Xmake: 3.0.9 (host B: 3.0.9, run with `XMAKE_ROOT=y` as the container is root)
- Python: 3.10.10
- PyTorch: 2.8.0+metax3.3.0.2

## Build

### NVIDIA

```bash
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
xmake f --nv-gpu=y -m release -c
xmake
xmake install -y
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e ./python
```

When building inside a root-owned container, set `XMAKE_ROOT=y` for the Xmake
commands above.

### MetaX

The MetaX build uses the MACA/MC toolchain and compiles the shared CUDA-like
kernels through MXCC:

```bash
export XMAKE_ROOT=y
export MACA_PATH=/opt/maca
export MACA_HOME=/opt/maca
export PATH=$HOME/.local/bin:/opt/conda/bin:/opt/maca/mxgpu_llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/mxdriver/lib:$LD_LIBRARY_PATH

xmake f --metax-gpu=y --use-mc=y --nv-gpu=n -m release -c
xmake
xmake install -y
python -m pip install -e ./python
```

The MetaX backend has its own device enum and Runtime API, while the operator
kernels are shared with NVIDIA through the vendor-neutral `src/ops/*/cuda` layer.

## Runtime and operator verification

```bash
python test/test_runtime.py --device nvidia

for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    python "test/ops/$op.py" --device nvidia
done

python test/test_qwen2_loader.py --device nvidia
```

All runtime and operator cases passed for Float32, Float16, and BFloat16. The
Qwen2 loader/reference test passed on NVIDIA, including incremental KV-cache
generation.

The same runtime suite, all eight operator suites, and the Qwen2 loader/KV-cache
reference suite passed on MetaX with `--device metax`. Coverage includes real
Qwen2 decode geometry (`12` query heads, `2` KV heads, head dimension `128`),
KV lengths `1`, `256`, and `257`, and the `151,936`-element argmax workload.

### The f32 reference was TF32, not f32

On a second C500 host the `linear` and `self_attention` f32 cases failed, by about
2e-4 relative — too coarse for fp32 on those shapes, which pointed at the
reference rather than the kernels. Torch defaults `allow_tf32` to True, so every
f32 matmul reference was running in TF32's 10-bit mantissa while the kernels
accumulate in fp32.

A float64 CPU computation arbitrates, since it is neutral between the two:

| case | our MACA kernel | torch on MACA | who is wrong |
| --- | --- | --- | --- |
| `self_attention` 2×2×1×1×4 f32 | 1.071e-08 | 1.397e-04 | the reference, by 13,000× |
| `linear` 512×4096×4096 f32 | 5.069e-06 | 3.394e-05 | the reference, by 6.7× |

The kernels were the accurate side in both cases; checking a 1e-8 result against a
1e-4 reference at a 1e-5 tolerance fails the kernel for the reference's error.
`test/test_utils.py` now disables TF32 for the whole suite, after which all eight
operator suites pass on metax; re-enabling it reproduces the failures, which
confirms the cause rather than assuming it. **The tolerances were not touched**, so
the tests are not weakened — the comparison is simply now against true fp32.

This was surfaced only by verifying on a second platform: this MACA build reports
`torch.backends.cuda.matmul.allow_tf32 == True`. Whether the NVIDIA host had the
same default was not re-checked — that host was no longer available — so the
harness now sets the flag explicitly rather than relying on any vendor's default,
which makes the suite correct on both platforms either way.

## End-to-end inference

```bash
python test/test_infer.py \
    --model /data/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --device nvidia \
    --test \
    --max_steps 128 \
    --prompt "Who are you?"
```

Result:

- Generated token sequence matched Transformers exactly.
- Transformers elapsed time: 2.09 seconds.
- LLAISYS elapsed time: 0.41 seconds.
- The previously observed 7,494 MiB process peak included the Transformers run
  in the same process and is not reported as an LLAISYS-only peak.

The KV cache is reserved from `input_tokens + max_new_tokens` instead of the
model's 131,072-token maximum context, preventing a fixed 3.5 GiB cache
allocation for short generations.

### Independent re-reproduction (both platforms)

The full pipeline was rebuilt from a clean tree and re-verified on fresh hosts
for both platforms:

- **NVIDIA RTX 4090 D** (driver 570.124.06, CUDA Toolkit 12.8, Ubuntu 24.04.1,
  Xmake 3.0.9, PyTorch 2.6.0): the `--nv-gpu=y` build completed in 10.2 seconds;
  runtime, tensor, all eight operator suites (Float32/Float16/BFloat16), and the
  Qwen2 loader all passed. The 128-step exact-match `test_infer.py --test`
  matched Transformers token-for-token, with Transformers at 2.12 seconds and
  LLAISYS at 0.41 seconds.
- **MetaX C500** (MACA 3.3.0.15, MX-SMI 2.2.9, driver 3.8.30, Ubuntu 24.04,
  Xmake 3.0.9, PyTorch 2.8.0+metax3.3.0.2): the
  `--metax-gpu=y --use-mc=y --nv-gpu=n` build completed in 12.8 seconds; the same
  runtime, tensor, eight operator, and Qwen2 loader suites all passed. The
  128-step exact-match `test_infer.py --test` matched Transformers token-for-token,
  with LLAISYS at 0.67 seconds.

Both runs reproduce the figures reported above from clean checkouts.

## NVIDIA optimization follow-up

- Qwen2 inference reuses a model-owned workspace instead of allocating hundreds
  of temporary GPU tensors for every generated token.
- K/V projections write directly into their cache slices, removing two
  synchronous device-to-device copies per layer and inference step.
- Single-token decoding uses fused attention without a global score buffer or
  per-layer `cudaMalloc`/`cudaFree`.
- CUDA argmax uses a block reduction instead of scanning the 151,936-element
  vocabulary with one thread. (This first version still used a single block; the
  two-stage rewrite is described under profile-guided optimization below.)
- Cache and workspace storage resize to the current request and release the old
  allocation before growing or shrinking, avoiding retained high-water memory
  and overlapping old/new allocations.

On the same RTX 4090, model, prompt, and 128-token exact-match test, observed
LLAISYS generation time decreased from 0.77 seconds to 0.41 seconds (46.8%).
The final token sequence still matched Transformers exactly.

Additional regression coverage includes the model's real decode geometry
(`12` query heads, `2` KV heads, head dimension `128`) at KV lengths `1`, `256`,
and `257`, a two-layer Qwen2 reference model, cache shrink/grow reuse, duplicate
maxima, NaNs, and the real `151,936`-element vocabulary.

## Correctness and API-robustness pass

Five issues were found by reading the tree against its own claims, and fixed
before any performance work, since each one either invalidated a test result or
made a failure unreportable.

| Issue | Why it mattered | Fix |
| --- | --- | --- |
| `test/ops/embedding.py` called `check_equal` without `assert` | `check_equal` returns a bool, so this test printed mismatches and still exited 0 — it was structurally incapable of failing, while the report claimed all operator cases passed | Added the `assert`, matching the other seven suites |
| `python/llaisys/runtime.py` had a debug `print` on every device free | Wrote a line to stdout per deallocation | Removed |
| No `try`/`catch` anywhere on the `extern "C"` boundary | Every exported function could throw (`CHECK_ARGUMENT` raises `std::runtime_error`); an exception unwinding through the C ABI into ctypes is undefined behavior, so one out-of-range token could abort the Python process instead of raising something catchable | Added an error-translation layer (below) |
| `generate()` accepted `temperature`/`top_p`/`top_k` and discarded them | The signature promised sampling; the backend is greedy-only, so callers silently got greedy decoding and could believe otherwise | Non-greedy values now raise `ValueError` naming the expected value |
| `ensure_workspace` compared capacity with `!=` | `ntoken` alternates between `prompt_len` and `1`, so every prefill/decode transition reallocated the entire workspace | Grows only when the request exceeds capacity, then slices to the actual length |

### C ABI error translation

`src/llaisys/error.{hpp,cc}` stores a thread-local message and wraps each entry
point in `protect()`; `include/llaisys/error.h` exposes `llaisysGetLastError`;
`python/llaisys/libllaisys/error.py` installs a ctypes `errcheck` that raises
`LlaisysError`. All 32 exported functions route through `protect()`, so a C++
exception becomes a Python exception instead of unwinding through the ABI.
Verified on device:

```
PASS view element mismatch -> tensorView: View shape must preserve the number of elements
PASS slice out of bounds   -> tensorSlice: Slice end is out of range
PASS unimplemented stub    -> llaisysRearrange: Unimplemented function
PASS shape mismatch add    -> llaisysAdd: Shapes mismatch
process still alive -> no UB abort
```

### Teardown during CUDA runtime unload

Every run printed `Failed to select device while destroying runtime: cudaSetDevice
failed: driver shutting down` at exit. The cause is ordering: the CUDA runtime
begins unloading before the thread-local `Context` is destroyed, so `cudaSetDevice`
from `Runtime`'s destructor necessarily fails.

Silencing the destructor's diagnostic would have hidden genuine leaks, so the
distinction is made at the device layer instead. CUDA reports this specific
condition as `cudaErrorCudartUnloading`, which means the runtime is already gone
and the resource the call would have released no longer exists — the call is a
successful no-op. `checkCudaTeardown` accepts only that status and defers
everything else to the normal throwing check, so it is applied to the four
teardown paths (`cudaSetDevice`, `cudaStreamDestroy`, `cudaFree`, `cudaFreeHost`)
while real failures still report. The message is gone and the runtime, loader,
and 128-step exact-match tests still pass.

The equivalent MACA change was deliberately not made: the MetaX host was not
reachable to confirm the enumerator's name, and guessing it would risk exactly
the kind of unverified-platform build break described under platform status.

That deferral has since been resolved on hardware, and the answer was that no
port should be made. Two facts settle it. MACA has no `cudaErrorCudartUnloading`
equivalent — `/opt/maca/include/mcr/mc_runtime_types.h` defines
`mcErrorDeinitialized = 4` and nothing named for runtime unloading — so the
guessed enumerator would not have compiled. More decisively, MetaX does not
exhibit the symptom: `test_runtime.py --device metax` and a Qwen2 generate both
exit cleanly with no teardown diagnostic and status 0. Adding a MACA teardown
exemption would therefore suppress a class of error that this platform reports
only when it is real. The asymmetry is recorded rather than papered over.

## Profile-guided optimization

### Measuring first: decode is memory-bound, not launch-bound

The intuitive read on 400+ kernel launches per token is that decode is
launch-bound. A decisive experiment — hold the launch count fixed and vary the
work per launch, by running a single forward pass at different `ntoken` — says
otherwise:

| ntokens | ms | ms/token |
| --- | --- | --- |
| 1 | 4.927 | 4.927 |
| 2 | 5.969 | 2.985 |
| 8 | 6.015 | 0.752 |
| 32 | 6.074 | 0.190 |
| 64 | 6.609 | 0.103 |
| 128 | 10.252 | 0.080 |

Cost is nearly flat to `ntoken=64`, so a pass is dominated by a fixed term, not
by arithmetic. That term is weight bandwidth: reading 3.56 GB of BF16 weights in
4.93 ms is **722 GB/s, or 72% of the 4090's 1008 GB/s peak**. Measured empty-kernel
launch latency is 2.36 µs, so 508 launches ≈ 1.2 ms, which hides behind 3.53 ms
of real GPU work.

This retired two planned optimizations before they were written — vectorizing
`add`/`swiglu` and overlapping streams both target a bottleneck that does not
exist here. It also identified the real targets, the two operators running far
below achievable bandwidth:

| op | achieved | cause |
| --- | --- | --- |
| `self_attention` (decode) | 128 KB / 28 µs = **4.5 GB/s** | 12 blocks on 128 SMs; each dot product recomputed across three passes (max, sum, output); GQA reread K/V per query head |
| `argmax` (151,936) | 296 KB / 105 µs = **2.8 GB/s** | launched `<<<1, 256>>>` — a single block |

At ~16% and ~2.1% of a decode step respectively, these were the only two
operators where kernel work, rather than weight traffic, set the cost.

### Kernel results

| op | before | after | change |
| --- | --- | --- | --- |
| `argmax` (151,936) | 104.66 µs | **6.20 µs** | 17× — two-stage reduction, `<<<blocks, 256>>>` then one folding block. NaN and tie-breaking semantics preserved; a single-block path avoids regressing small inputs |
| `rope` | 8.6 µs | 4.65 µs | 1.85× — one block per token computes each `(sin, cos)` pair once into shared memory instead of a `powf`+`sincosf` per output element |
| `embedding` | 7.8 µs | 5.11 µs | 1.53× — one block per row, coalesced gather, **plus an out-of-bounds fix** (see below) |
| `self_attention` (decode) | see curve | see curve | flash-decoding split-KV |

The `embedding` change fixed a real bug, not just throughput. Indices live in
device memory, so validating them on the host would need a device-to-host sync on
the decode path; the old kernel therefore cast whatever it read to `size_t` and
gathered, meaning a negative or out-of-range token id read out of bounds. The
kernel now range-checks per row and zero-fills invalid rows. The out-of-bounds
read was reproduced and confirmed fixed on device.

### Flash-decoding split-KV attention

`self_attention` took three attempts, and the first two are the useful part of
the record. A single-pass rewrite and then a warp-shuffle reduction both left the
op at ~6.8 GB/s *regardless of KV length* — a flat ceiling is the signature of an
occupancy limit, not a memory or reduction one. The root cause was the launch
geometry: one block per query head is **12 blocks on 128 SMs**, so 90% of the GPU
sat idle no matter how efficient the block became.

The fix is flash-decoding: split the key range into 512-element chunks so the
grid becomes `query_heads * splits`, have each chunk emit its unnormalized
partial output alongside its softmax statistics, and merge with the standard
log-sum-exp rescale. At `kv=16384` that is 32 splits × 12 heads = 384 blocks.
Below `kv=1024` the combine pass costs more than it saves, so the single-block
decode kernel is kept for short contexts. A thread-local workspace holds the
partials and grows monotonically instead of allocating per call.

Correctness was checked against `torch.nn.functional.scaled_dot_product_attention`
across the threshold, confirming both paths and that the switch is not a
discontinuity:

```
kv=1023   split_path=False  match=True
kv=1024   split_path=False  match=True
kv=1025   split_path=True   match=True
kv=1536   split_path=True   match=True
kv=2048   split_path=True   match=True
kv=3000   split_path=True   match=True
kv=4096   split_path=True   match=True
kv=8192   split_path=True   match=True
```

### End-to-end decode throughput vs context

Attention is the one part of decode that grows with context, so a short-prompt
benchmark hides it: at `kv=90` attention is ~13% of a token and a faster kernel
barely moves the total. `test/benchmark_context.py` sweeps the starting context.
Baseline here means this tree with the four pre-optimization kernels restored, so
the comparison isolates the kernel work from every other change:

| context | baseline ms/token | optimized ms/token | speedup |
| --- | --- | --- | --- |
| 128 | 5.442 ± 0.294 | 5.273 ± 0.091 | 1.03× |
| 512 | 7.400 ± 0.161 | 6.713 ± 0.091 | 1.10× |
| 1024 | 9.782 ± 0.324 | 6.597 ± 0.356 | 1.48× |
| 2048 | 14.667 ± 0.225 | 7.244 ± 0.080 | 2.02× |
| 4096 | 25.673 ± 0.257 | 8.461 ± 0.247 | 3.03× |
| 8192 | 47.622 ± 0.576 | 8.455 ± 0.229 | 5.63× |
| 16384 | 87.305 ± 1.300 | 8.192 ± 0.626 | **10.66×** |

Baseline doubles its per-token cost with every doubling of context, as expected
when attention is serialized on 12 blocks. The optimized curve is flat from 4096
onward — 8.46, 8.46, and 8.19 ms/token at 4096, 8192, and 16384 agree within
their uncertainties — so decode has returned to being bounded by weight
bandwidth rather than by attention.

### Benchmark methodology

The first version of this benchmark reported 0.642 ms/token at context 16384, an
11× *speedup* over its own 8192 result. That is physically impossible, and the
cause was the measurement, not the kernels: per-step cost was recovered as
`median(total) - median(prefill)`. At context 16384 prefill is ~29.8 s while 31
decode steps are ~0.3 s, so the signal is 1% of the subtrahend and prefill jitter
alone exceeds the entire quantity being measured. The difference went negative
(-3.9 ms/token) and a `max(..., 0)` clamp turned that into a small positive
number that looked like a fast reading.

The benchmark now differences two generation runs of different length
(`steps` and `2 * steps`) instead. Both runs pay the same prefill, so it cancels
algebraically rather than numerically. Three guards make a future failure of this
kind loud instead of silent:

- Actual tokens generated is compared against tokens requested, so an EOS or
  cache-limit truncation raises instead of averaging over steps that never ran.
- Run-to-run spread is propagated and reported as `±` on every number.
- A difference smaller than 3× the observed spread raises `UnreliableMeasurement`
  rather than being reported. This fired at context 16384 with `--steps 32`,
  which is why the long-context rows above were measured with `--steps 256`.

Cross-checking the two methods on the same host shows the old approach was sound
where the decode signal was large relative to prefill and degraded exactly where
theory predicts:

| context | prefill-subtraction | differencing | error |
| --- | --- | --- | --- |
| 128 | 5.230 | 5.302 | 1.4% |
| 1024 | 6.561 | 6.619 | 0.9% |
| 4096 | 8.429 | 8.368 | 0.7% |
| 8192 | 9.241 | 8.218 | 12% |
| 16384 | -3.878 | 9.676 | invalid |

Conclusions at or below 4096 were unaffected; only the 8192 and 16384 rows
needed re-measurement.

### Supporting tools

- `test/benchmark_infer.py` — prefill and decode reported separately, with
  repeats and JSON output for before/after diffs.
- `test/benchmark_ops.py` — per-operator microseconds at Qwen2 decode geometry,
  with calls-per-token, to attribute a decode step across operators. `ncu` needs
  GPU performance counters that containers typically withhold, so cost is
  attributed by timing operators in isolation instead.
- `test/benchmark_context.py` — the decode-throughput-vs-context sweep above.

### The bottleneck is not the same on both platforms

Everything above was measured on the 4090, where profiling showed decode to be
weight-bandwidth-bound and explicitly *retired* kernel-launch count as a target.
Re-running the same experiments on MetaX C500 (32 GB sGPU quota, 50% compute)
shows that conclusion is platform-specific, and inverts there.

The per-launch cost differs by 2.8×. Batching N one-element `add` calls behind a
single synchronization and dividing gives the marginal cost of a launch:

| platform | per-launch | × ~508 launches/token | single-token pass | launch share |
| --- | --- | --- | --- | --- |
| RTX 4090 | 2.36 µs | 1.20 ms | 4.93 ms | 24% (hidden under 3.53 ms of work) |
| MetaX C500 | 6.54 µs | 3.32 ms | 6.56 ms | **51%** |

On the 4090 the host stays comfortably ahead of the device, so launch count is
free. On C500 more than half of a decode step is launch overhead, so **launch
count is the dominant cost there** — the optimization that measurement correctly
rejected for NVIDIA is the main opportunity for MetaX. Kernel fusion and CUDA/MC
graph capture, both no-ops on the 4090, would target this directly.

The same batching sweep exposes the floor per call:

```
add (1x1536 bf16), N calls behind one sync — per-call µs on C500
N=1   40.46      N=8   17.48      N=64   7.55      N=256  6.80
```

A single timed call costs 40 µs, but the marginal cost converges to ~6.8 µs,
matching the launch overhead above. `benchmark_ops.py` times one call per sample
and so reports the 40 µs figure; its modelled total of 17.9 ms/token overstates
the measured ~4.8 ms/token by 3.7× on this platform. **The harness is sound on
NVIDIA and misleading on MetaX**, because it assumes per-call latency is small
relative to kernel time. It is used here only for relative op ranking, and the
end-to-end numbers below come from `benchmark_context.py`.

Flash-decoding does carry over. The context sweep at `--steps 128 --repeats 3`:

| context | ms/token | ± | tok/s |
| --- | --- | --- | --- |
| 128 | 10.184 | 0.017 | 98.2 |
| 512 | 15.466 | 2.032 | 64.7 |
| 1024 | 12.423 | 1.044 | 80.5 |
| 2048 | 12.466 | 3.681 | 80.2 |
| 4096 | *unreliable* | — | — |

Cost is flat from 1024 to 2048 (12.42 vs 12.47, well inside the spread), which is
the same signature the 4090 showed once attention stopped serializing — so the
split-KV kernel is doing its job on MACA without retuning, despite the 512-element
split and `MAX_BLOCKS = 256` having been chosen for a 128-SM NVIDIA part.

Two honest caveats. This sGPU is far noisier than the dedicated 4090 (±3.7 ms at
2048, versus ±0.2 ms for the same measurement on NVIDIA), and the 512 row sits
above the 1024 row, which is not physical — with a ±2.0 ms spread the two rows
overlap, so the ordering is noise rather than a regression. The 4096 row was
withheld entirely: the `UnreliableMeasurement` guard fired because the 1568 ms
signal did not clear 3× the 1186 ms spread. That guard doing its job on a second,
noisier platform is itself the useful result — the alternative was a fabricated
number of exactly the kind the methodology section above was written to prevent.
Establishing a baseline-vs-optimized ratio on C500 needs either a dedicated card
or many more repeats, and is not claimed here.

## MetaX end-to-end verification

```bash
python test/test_infer.py \
    --model /data/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --device metax \
    --test \
    --max_steps 128 \
    --prompt "Who are you?"
```

On the C500 sGPU, the token sequence from the 128-step test matched Transformers exactly.
Transformers took 2.88 seconds and LLAISYS took 0.68 seconds. Peak sGPU VRAM
across the sequential reference and LLAISYS runs was 4,160 MiB; the allocation
returned to 0 MiB when the process exited. A separate LLAISYS-only run peaked at
3,814 MiB and also returned to 0 MiB. The same combined-process peak was observed
for a 16-token safety run, confirming that model weights dominate and the
request-sized KV cache avoids the previous fixed-context out-of-memory behavior.

After the vendor-neutral CUDA-layer and runtime-lifetime refactor, the complete
MetaX build, Runtime test, eight operator suites, Qwen2 loader, and 128-step
exact-match test were rerun from a clean directory. The reference and LLAISYS
runs took 2.79 and 0.68 seconds in the final rerun. A monitored run raised
system VRAM from 846,592 KiB to 5,106,624 KiB (a 4,160 MiB delta); memory
returned to baseline after exit, and `mx-smi` reported no remaining process.

## CPU regression

The Runtime lifetime regression, tensor test, all operator tests, and the Qwen2
loader/reference test passed in a clean CPU-only build after the architecture
refactor.
