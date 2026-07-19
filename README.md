# nnLive

**Faithful, browser-native [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) 3D promptable segmentation — running entirely in your browser on WebGPU.**

nnInteractive is a state-of-the-art 3D promptable segmentation model (MIC-DKFZ). nnLive distills it into a compact student and runs it through a hand-written WebGPU (WGSL) runtime, so you can click-to-segment volumetric CT/MR/PET **with no install and no server** — the model executes on your GPU, in a tab.

### ▶ Live demo — **[pieper.github.io/nnLive](https://pieper.github.io/nnLive/)**

Spin a random case from the NCI [Imaging Data Commons](https://imaging.datacommons.cancer.gov/) and click to segment. Requires a WebGPU browser (Chrome/Edge 113+, Safari 18+) with `shader-f16`.

---

## What "faithful" means

nnLive reproduces nnInteractive's actual interaction protocol, verified against the shipped inference code:

- **8-channel input** `[image, prev_seg, bbox±, point±, scribble±]`, early-fused.
- **Global whole-volume z-score** normalization; patches cropped from the normalized volume, out-of-bounds padded with 0.
- **Point prompts = smooth Euclidean-distance-transform balls, radius 4**, unioned by max.
- **Interaction decay 0.9** — older prompts of a category fade as new ones are added.
- **Autoregressive refinement** — the previous prediction is fed back through the `prev_seg` channel, so every click *refines the running mask* instead of starting over.
- **Auto-zoom ×1.5→×4** — when the object exceeds the 192³ field of view, the patch zooms out and re-predicts, then refines at native resolution.

The student is distilled from the nnInteractive teacher with a **true autoregressive rollout** (the teacher's own predictions become the `prev_seg` context and corrective clicks are sampled from the teacher-vs-label error), so it learns to refine — not just to predict a first guess.

## How it runs in the browser

- **Path A decomposition** — the frozen image trunk (encode-once per patch) is split from the prompt-conditioned decoder (decode-per-click), preserving interactive latency.
- **Custom WGSL runtime** (`docs/js/wgpu-net.js`) — a minimal ONNX-graph executor over hand-written fp16 WebGPU kernels (Conv / InstanceNorm / LeakyReLU / Add / Pool / Resize). No ONNX Runtime, no WASM.
- The demo reports **your GPU and the expected per-click latency**, which varies widely across hardware (sub-second on a discrete GPU, several seconds on integrated graphics).

## `idc_tools`

A standalone **TypeScript utility library** (`idc_tools/`) to stream and reconstruct imaging series straight from the IDC public buckets — including **SEG Roulette** (spin a random annotated series). No server; DICOM is decoded in a worker.

```ts
import { spinRandom } from 'idc_tools';
const { ct, seg, entry } = await spinRandom(
  { onProgress: p => console.log(p.msg) },
  { manifestUrl: 'segroulette.json', filter: e => e.m === 'CT' });
// ct.vol (Int16 HU), ct.dims [nx,ny,nz], ct.ijkToRAS, ct.win/lev ; seg?.lab on the same grid.
```

Build: `cd idc_tools && npm install && npm run build`. It will eventually get its own repository.

## Repository layout

```
docs/            GitHub Pages site (the live demo) — index.html + js/ + idc_tools/ + models/
idc_tools/       standalone TypeScript library (IDC streaming + SEG Roulette)
training/        distillation scripts (teacher-in-the-loop, faithful autoregressive rollout)
livemodule/      development probes and the broader place-agnostic LiveModule design
```

nnLive is also a component of **SlicerLive**: the `livemodule/` half is a place-agnostic inference API + broker that runs wherever the data allows (browser GPU, local helper app, or remote GPU). See [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md) for that roadmap.

## Licensing

- **Code** (this repository, including `idc_tools` and the WebGPU runtime): **Apache License 2.0** — see [`LICENSE`](LICENSE).
- **Model weights** (distributed as GitHub release assets): the student is derived from nnInteractive and therefore **inherits nnInteractive's [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) terms — non-commercial, share-alike, with attribution.** See [`MODELS.md`](MODELS.md).
- **Imaging data** is streamed live from the IDC public buckets at runtime and is **not redistributed by this repository**; each IDC collection carries its own license (shown in the viewer).

## ⚠️ Not for clinical use

nnLive is a research and educational demonstration. It is **not a medical device** and must not be used for diagnosis or treatment.

## Credits

- **[nnInteractive](https://github.com/MIC-DKFZ/nnInteractive)** — Isensee, Rokuss, et al., MIC-DKFZ. The teacher model and interaction design.
- **[NCI Imaging Data Commons](https://imaging.datacommons.cancer.gov/)** — the open imaging streamed by the demo.
- Thanks to **Andrey Fedorov** for testing and feedback that shaped the SEG Roulette experience.
