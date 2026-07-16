# Capability manifest

The manifest is the contract the **broker** routes on. It is a static JSON document published alongside
the weights (on the CORS Ceph bucket) describing the available **model variants**, their **device
requirements**, and where to **fetch** them. The broker combines this with a **device probe** and a
**per-dataset fit analysis** to choose a `ComputeMode` + variant and to render honest tradeoff numbers.

See [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md) §2 for how it's used.

## Shape (draft)

```jsonc
{
  "schemaVersion": "0.1",
  "updated": "2026-07-16",
  "license": "CC-BY-NC-SA-4.0",           // student weights are NC-SA derivatives
  "attribution": "Distilled from nnInteractive (DKFZ), arXiv:2503.08373",

  "variants": [
    {
      "id": "nnlive-96",
      "patch": [96, 96, 96],
      "backbone": "ResEnc-M",
      "inputChannels": 8,
      "precision": "fp16",
      "weights": { "onnx": "https://<ceph>/nnlive/nnlive-96.onnx", "bytes": 62000000 },
      "requires": {                         // what the device probe must satisfy
        "webgpu": true,
        "shaderF16": true,
        "maxStorageBufferBindingSize": 134217728,   // fits under the 128 MiB DEFAULT
        "approxGpuMemoryMB": 500
      },
      "deviceTier": "universal",            // phones, iGPU, discrete, Apple
      "domains": ["ultrasound", "echo", "mr-small-structures"],
      "eval": "https://<ceph>/nnlive/eval/nnlive-96.json"   // student-vs-teacher curves
    },
    {
      "id": "nnlive-128",
      "patch": [128, 128, 128],
      "backbone": "ResEnc-M",
      "requires": { "webgpu": true, "shaderF16": true,
                    "maxStorageBufferBindingSize": 268435456, "approxGpuMemoryMB": 1200 },
      "deviceTier": "mainstream",
      "weights": { "onnx": "https://<ceph>/nnlive/nnlive-128.onnx", "bytes": 110000000 }
      // ...
    },
    {
      "id": "nnlive-160",
      "patch": [160, 160, 160],
      "requires": { "webgpu": true, "shaderF16": true,
                    "maxStorageBufferBindingSize": 2147483648, "approxGpuMemoryMB": 2300 },
      "deviceTier": "discrete-or-apple"
      // ...
    },
    {
      "id": "teacher-192",
      "patch": [192, 192, 192],
      "backbone": "ResEnc-L",
      "computeModes": ["LocalHelper", "RemoteGPU"],   // not offered InBrowser
      "canonical": true,                               // the "promote / finalize" target
      "requires": { "approxGpuMemoryMB": 6000 }
    }
  ],

  "servers": [                              // optional pre-registered RemoteGPU endpoints
    { "id": "modal-default", "url": "https://<modal-app>.modal.run", "authHeader": "x-api-key",
      "gpu": "A10G", "capabilities": ["nninteractive", "totalsegmentator-batch"] }
  ]
}
```

## Device probe → variant selection (broker, client-side)

1. `navigator.gpu.requestAdapter()`; read `adapter.limits`.
2. Attempt `requestDevice({ requiredLimits, requiredFeatures: ["shader-f16"] })` at increasing tiers.
3. Run a **reference forward** for the largest variant whose `requires` are met; time it.
4. Classify: `fast` (< ~2 s/patch) / `slow` (interactive-with-patience) / `unavailable`.
5. Pick the largest variant that is `fast`; expose the quality/latency slider down to `nnlive-96`.
6. Cache the result keyed by adapter identity.

## Per-dataset fit analysis (broker, per volume)

Given `dims`, `spacing`, `targetBox` and the chosen variant `patch`:
- `fovMM = patch * chosenSpacing` → does `targetBox` fit one patch?
- `autoZoomPasses = ceil(targetExtent / fovMM)` → predicted latency = `passes * perPatchLatency`.
- `renderBytes = prod(resampledDims) * bytesPerVoxel` → fits the render/storage budget?
- Emit the three-column comparison (InBrowser / LocalHelper / RemoteGPU) in real numbers.

## Provenance record (attached to each produced segmentation)

```jsonc
{ "producedBy": "nnlive-96", "computeMode": "InBrowser", "canonical": false,
  "promoteTarget": "teacher-192", "interactionLog": "<hash>" }
```

`interactionLog` is the accepted prompt sequence — replaying it through `teacher-192` on a
LocalHelper/RemoteGPU backend produces the canonical result ("finalize on full model").
