# nnLive — cross-platform interactive segmentation for SlicerLive

**nnLive** brings [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive)-style 3D promptable
segmentation to the web, running *wherever the data allows* — in the browser on the client GPU, in a
local helper app, or on a remote GPU — behind one inference API that hides where the work happens.

It has two halves:

1. **`training/`** — a **knowledge-distillation** pipeline that turns the full 192³ nnInteractive
   ResEnc-L "teacher" into a family of smaller-patch **student** models (96³ / 128³ / 160³) that fit
   the browser's WebGPU memory budget — down to phones for the smallest. The teacher is an unlimited,
   self-labeling data generator, so training converges fast and the accuracy tradeoffs are measured
   automatically (student-vs-teacher), not guessed.

2. **`livemodule/`** — the **LiveModule**: a place-agnostic inference API plus a **broker** that
   probes the running device, examines the data, and is transparent about what you can do *here*
   (in-browser) versus with a *helper app* or a *server* — exposing crop/resample choices the user
   decides on, against real per-device latency and accuracy numbers.

nnLive is a component of **SlicerLive** (`~/slicer/SlicerLive`) and implements the `LiveModule`
role from its [architecture](../SlicerLive/docs/ARCHITECTURE.md) §3.3: a classic module recast as a
service that speaks the LiveScene protocol and computes wherever its compute lives.

## Repository map

| Path | What |
|---|---|
| [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md) | **The plan** — findings, architecture, phased roadmap, cost/time budgets |
| [`docs/CAPABILITY-MANIFEST.md`](docs/CAPABILITY-MANIFEST.md) | The model-variant + device-class manifest the broker routes on |
| `training/` | Distillation trainer, patch-size plans, automatic student-vs-teacher eval |
| `livemodule/` | Browser inference API, broker (device probe + fit analysis), backends |
| `server/` | Reference GPU server (reuses the existing nnInteractive Modal endpoint) |
| `manifest/` | Published capability manifest + example weight entries |

## Status

Bootstrapping (2026-07-16). Start with [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md).

## Licensing

Code in this repo is **Apache-2.0**. **Student model weights are derivatives of the nnInteractive
weights (CC-BY-NC-SA 4.0) and therefore inherit non-commercial + share-alike terms** — see
[`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md) §10. Published weight artifacts must carry that
license explicitly.
