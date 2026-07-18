# Custom WGSL per-click inference pipeline (nnInteractive → browser, client-native)

A from-scratch WebGPU (WGSL) inference runtime that runs the **Path A** per-click network
(distilled nnInteractive: frozen image trunk + prompt injectors + deep stages + slim decoder)
**without ONNX Runtime**, to escape ORT-Web's naive/unoptimized 3D-conv WGSL kernel.

## Why
ORT-Web 1.25+ ships a *naive* per-output-element Conv3D WGSL kernel (PR #27917), ~5–7× slower than
its own WASM/XNNPACK on 3D CNNs. All ORT WebGPU momentum is on transformer/LLM ops; WebNN has no
conv3d and no Safari support. So we write our own kernels. Goal: **client-native, cross-platform,
hardware-limit-matched** interactive segmentation (vs nnInteractive's NVIDIA-discrete-GPU-only server).

## Results (Apple M4, headless, per-click, fp16, validated 99.98% mask sign-agree vs ORT)
| patch | ORT-WebGPU | WASM (XNNPACK) | **custom WGSL** |
|---|---|---|---|
| 64³  | 3.5 s | 0.66 s | **0.30 s** |
| 128³ | 36 s  | 5.4 s  | **2.0 s** (18× vs ORT, 2.7× vs WASM) |

Custom Conv3D kernel: naive 189 GF/s (4% peak) → vec4 register-blocked **1084 GF/s (27% of M4 peak)**.
Projected discrete-GPU 128³/192³: sub-second.

## Files
- `wgpu-net.js` — runtime: device init, fp16<->f32, pipeline cache, and WGSL kernels:
  Conv3D (V4 vec4 register-blocked; generalized kernel∈{1,3}, stride∈{1,2}, pad, bias),
  InstanceNorm3D (stats + apply, + fused `IN_APPLY_LEAKY`), LeakyReLU, Add, Avg/Max-Pool
  (generalized kernel/stride), Resize (nearest/floor).
- `wgpu-net-test.html` — minimal ONNX-graph executor: loads `*.graph.json` + `*.weights.bin`,
  runs all nodes in one batched command encoder, validates vs ORT ref + times.
  `?p=64|128 &notest=1 (timing only) &dbg=1 (per-node diff bisection)`.
- `conv3d-bench.html` — standalone Conv3D kernel benchmark (naive vs tiled vs V4/V8/Vsm), `?prec=f16|f32`.
- `wgpu-block-test.html` — Conv→InstanceNorm→LeakyReLU block validation (runtime nucleus).
- `wgsl/dump_graph.py` — perclick ONNX → `graph.json` (231 nodes, shapes resolved, shape-math dropped) + f16 `weights.bin`.
- `wgsl/gen_ref.py` / `gen_ref_debug.py` — ORT reference (+ per-node intermediates) for validation.
- `wgsl/export_pathA.py` — export the trained Path A model (r2 + slim decoder) to ONNX (trunk / perclick / full), fp16.

## Regenerate the (git-ignored) binaries
```
# needs the trained checkpoints (pathA_ckpt/) + nnInteractive weights
python wgsl/export_pathA.py 128          # -> perclick_128.onnx etc.
python wgsl/dump_graph.py  perclick_128.onnx   # -> perclick_128.graph.json + .weights.bin
python wgsl/gen_ref.py     perclick_64.onnx    # -> models/pathA/ref/*.f32
# serve + run headless:  python serve.py 8801 ; chrome --headless=new --enable-unsafe-webgpu ... wgpu-net-test.html?p=128
```

## Remaining optimization headroom
adaptive conv (V4 high-channel / Vsm low-channel decoder); buffer pooling (192³ + constrained devices);
more fusion (conv-bias, decoder up+lateral add); wire trunk (encode-once) + a `custom-wgsl` demo backend.
