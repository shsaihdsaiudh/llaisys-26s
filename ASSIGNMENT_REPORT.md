# LLAISYS Assignment Reproduction Report

## Platform status

| Platform | Runtime | Operators | Qwen2 inference | Status |
| --- | --- | --- | --- | --- |
| CPU (x86_64) | Passed | Passed | Passed | Supported |
| NVIDIA RTX 4090 | Passed | Passed | Passed | Supported |
| MetaX C500 | Passed | Passed | Passed | Supported |

Operator coverage includes Float32, Float16, and BFloat16 on every platform. The
Qwen2 pipeline (loader, KV-cache decode, and the 128-step exact-match generation
test) passes on all three backends.

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

- GPU: MetaX C500, 16 GB sGPU quota, 25% compute quota
- Driver: 3.8.30
- MACA: 3.3.0.15
- MXCC: 1.0.0
- OS: Ubuntu 24.04.1 LTS
- Xmake: 3.0.9
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
- CUDA argmax uses a 256-thread block reduction instead of scanning the
  151,936-element vocabulary with one thread.
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
