# livemodule/ — cross-platform inference + broker

The **LiveModule**: a place-agnostic inference API and the **broker** that picks where inference runs
(browser / local helper / remote GPU) and is transparent about the tradeoffs. Implements the
`LiveModule` role from SlicerLive's [architecture](../../SlicerLive/docs/ARCHITECTURE.md) §3.3.
See [`../docs/DEVELOPMENT_PLAN.md`](../docs/DEVELOPMENT_PLAN.md) §2 and
[`../docs/CAPABILITY-MANIFEST.md`](../docs/CAPABILITY-MANIFEST.md).

## Inference API (place-agnostic)

```
createSession(volumeRef, {spacing, cropBox, modelVariant}) -> sessionId
interact(sessionId, prompt) -> maskDelta        // point / bbox / scribble / lasso, ±
reset(sessionId) / finalize(sessionId) -> segmentationNode
```

Results write back as `vtkMRMLSegmentationNode`s over SlicerLive's node-state/blob channels; the caller
never learns where inference happened.

## Backends (`ComputeMode`)

| Mode | Transport | PHI | Notes |
|---|---|---|---|
| `InBrowser` | WebGPU (ORT-Web) + student variant | stays in tab | desktop discrete/Apple; 96³ on iGPU/phones |
| `LocalHelper` | loopback HTTP (desktop/Android) / iOS embedded WebView | stays on machine | PHI-safe default; full 192³ teacher |
| `RemoteGPU` | HTTPS (Modal/vast/JS2/cloud) | leaves machine | reuses `../server/modal_app.py` |

## The broker

- **Device probe** (once, cached): raised-fp16-limits request → reference forward → latency class →
  largest viable variant.
- **Fit analysis** (per dataset): FOV / AutoZoom passes / render budget → live crop/resample readout +
  three-column comparison.
- **Provenance + promote**: tag masks with the producing variant; "finalize on full model" replays the
  interaction log through the 192³ teacher.

## Planned layout

```
livemodule/
├── src/
│   ├── api.ts           # createSession / interact / reset / finalize
│   ├── broker.ts        # device probe + per-dataset fit analysis + variant selection
│   ├── backends/
│   │   ├── inbrowser.ts # ORT-Web WebGPU session + AutoZoom orchestration
│   │   ├── localhelper.ts
│   │   └── remotegpu.ts
│   └── session.ts       # ±channel encoding, target-buffer accumulation (ported from server)
└── test/
```
