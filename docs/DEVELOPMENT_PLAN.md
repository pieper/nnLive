# nnLive — development plan

*Founding plan, 2026-07-16. Distills a design conversation into an executable roadmap. Companion to
SlicerLive's [`ARCHITECTURE.md`](../../SlicerLive/docs/ARCHITECTURE.md) — nnLive is the concrete
`LiveModule` (§3.3) for interactive segmentation.*

---

## 0. Thesis

Interactive 3D segmentation (nnInteractive) can run **in a mainstream browser** for the small-volume,
high-information-content cases that matter clinically — ultrasound, cardiac/prostate/musculoskeletal
MR, echo, lymph nodes and other smallish structures — and can **escalate gracefully** to a local
helper app or a remote GPU when the data or the device demands it. One inference API serves all three;
a **broker** picks the backend and is honest, in real numbers, about the tradeoff.

Two enabling bets:

1. **The model is retrainable to the platform.** nnInteractive's cost is fixed by its **192³ patch**.
   We distill smaller-patch **students** whose per-click memory and latency fit the browser — down to a
   96³ variant that runs on almost anything, phones included, with no special WebGPU limits.
2. **The teacher labels the students for free.** The full 192³ ResEnc-L model generates unlimited
   (image, prompt, mask) training data, so distillation converges in GPU-days and the accuracy
   tradeoffs are *measured* (student-vs-teacher), not asserted.

---

## 1. Established findings (the technical basis)

### 1.1 nnInteractive is a fixed-patch model → inference cost is constant

- Backbone **ResEnc-L nnU-Net**, patch **192×192×192**, ~8 input channels (image + point±/bbox±/
  scribble±/mask/previous-prediction guidance). Recommends **10 GB VRAM (<6 GB for small objects)**.
  ([paper](https://arxiv.org/abs/2503.08373), [repo](https://github.com/MIC-DKFZ/nnInteractive)).
- The network always runs at its trained patch, **cropped around the prompt** — so a 200³ echo volume
  and a whole-body CT cost the *same per click*. **Volume size never enters the forward pass.**
- Consequence: **"can this browser run nnInteractive?" is one per-device capability check, not a
  per-dataset question.** If a device can run one patch forward pass, *every* job is feasible on it.
- Crop/resample do **not** reduce inference cost. They control (a) the **physical field-of-view** the
  fixed patch covers (spacing → FOV: 0.6 mm ⇒ 115 mm at 192³), (b) whether the whole volume fits the
  **render/storage** budget, and (c) how many **AutoZoom** passes a large target needs.

### 1.2 Browser memory math (per 192³ patch, fp16)

| Item | Size |
|---|---|
| Input stack (~8 ch) | ~113 MB |
| First full-res 32-feature tensor | **453 MB** (single tensor) |
| Peak forward-pass working set | **~2–4 GB** |

Against WebGPU on Chrome/Edge, three **probeable gates** for the stock 192³ model:

1. A 453 MB tensor exceeds the default `maxStorageBufferBindingSize` (128 MiB) 3.5×. Must request a
   **raised limit** (desktop adapters report 2–4 GiB and grant it on discrete GPUs / Apple Silicon;
   weak iGPUs may refuse).
2. **fp16 mandatory** (`shader-f16`, Chrome 121+/Edge 122+; universal on modern discrete/Apple/AMD).
3. **~3–4 GB free GPU memory.**

Latency, one 192³ pass, mid discrete GPU / Apple Silicon: **~0.3–2 s**; AutoZoom ×2–3 for large
targets ⇒ ~1–5 s/click. Weak iGPU: 10–30 s (not interactive → broker should say so).

### 1.3 Patch size is the master dial (fp16 peak)

| Patch | Voxels | First tensor | Peak | Runs on |
|---|---|---|---|---|
| 192³ (stock) | 7.08 M | 453 MB | ~2–4 GB | discrete/Apple, **raised** limits |
| 160³ | 4.10 M | 262 MB | ~1.2–2.3 GB | mid GPUs, raised limits |
| 128³ | 2.10 M | 134 MB | ~0.6–1.2 GB | modest iGPU, small/no raise |
| **96³** | 0.88 M | **57 MB** | ~0.3–0.5 GB | **almost anything incl. phones — no raised limits** |

The 96³ row crosses *under* the 128 MiB default binding → runs in a **stock WebGPU context**, the
widest-compatibility tier (mobile Chromium, weak iGPUs). This is the browser-only baseline.

### 1.4 The one engineering prerequisite (spike this first)

nnInteractive ships **PyTorch weights, not ONNX/tflite.** Each variant must be **exported to ONNX and
validated to run entirely on the ORT-Web WebGPU EP** — a single unsupported 3D op silently falls back
to CPU/WASM and collapses latency. This gate is moot-until-proven; it is Milestone 0.

### 1.5 Tradeoffs of smaller students, and why they're cheaper than they look

- Cost of a smaller patch/narrower net: **less spatial context** ⇒ worse on large, low-contrast, or
  context-dependent structures; more resample/AutoZoom for big targets. Small US/MR targets lose least.
- **Interactive loops forgive capacity:** a weaker model reaches the *same final mask* with 1–2 extra
  clicks — the human closes the gap each iteration. "Worse Dice per click" partly washes out per session.
- **Distillation ≫ retrain-from-scratch:** train students to mimic the teacher's outputs; recovers most
  accuracy at a fraction of data/compute; can be scoped to target modalities.

### 1.6 Compute budget (established estimates)

Distillation GPU-time (A100-equivalent), driven by **breadth of anatomy/modality**, not student size:

| Deliverable | A100-hrs |
|---|---|
| Narrow-domain single variant (spike) | 24–72 (1–3 days) |
| General-purpose broad variant | 168–336 (1–2 weeks) |
| Family of 3 (narrow, shared pipeline) | 72–216 |

Hardware notes: 3D conv is **bandwidth-bound**, so H100 ≈ 1.5–2× A100 (not 3×); 4×H100 collapses the
family to **~2–4 days**, best used **one-variant-per-GPU** (embarrassingly parallel, avoids large-batch
LR retuning, preemption-robust). Likely bottleneck at scale is **data loading / Ceph I/O**, not compute
— precompute a teacher-label cache to local NVMe.

Cloud cost on **GCP Spot** (2026: A100 ~$1.10–1.50, H100 ~$2.5–3.7, L4 ~$0.21–0.35 /GPU-hr):

| Deliverable | A100 Spot | Cheapest (L4 Spot, small students) |
|---|---|---|
| Derisking spike | **$30–110** | $20–90 |
| General-purpose variant | **$185–500** | impractical (24 GB) |
| Family of 3 | **$80–320** | $60–260 |

A100 Spot is the cost sweet spot; H100 Spot buys wall-clock, not savings. **JS2 is $0 (research
allocation)** — GCP Spot is paid elastic overflow. At these numbers the choice is availability, not
budget. Spot requires **checkpoint + auto-resume** (nnU-Net checkpoints per epoch; budget ~10–20%
preemption overhead).

---

## 2. Architecture — the LiveModule and the broker

### 2.1 Place-agnostic inference API

One contract the viewer calls, identical regardless of backend (this is the shape the existing
nnInteractive server already exposes):

```
createSession(volumeRef, {spacing, cropBox, modelVariant}) -> sessionId   // upload/resample once
interact(sessionId, prompt) -> maskDelta                                   // point/bbox/scribble/lasso
reset(sessionId) / finalize(sessionId) -> segmentationNode
```

Inputs/outputs are **LiveScene nodes + content-addressed blobs**; results write back as
`vtkMRMLSegmentationNode`s over SlicerLive's node-state/blob channels. The caller never learns where
inference happened. Prompts are tiny (a point = a coordinate; scribble/lasso = a gzipped mask of just
the drawn region), so the interactive channel stays cellular-network-friendly.

### 2.2 Three backends behind the contract (`ComputeMode`)

| `ComputeMode` | Transport | PHI posture | Typical use |
|---|---|---|---|
| **InBrowser** | WebGPU (ORT-Web), student variant | data never leaves tab | US/MR/echo small structures, desktop discrete GPU / Apple Silicon; 96³ tier on iGPU/phones |
| **LocalHelper** | loopback HTTP (`localhost`) / iOS embedded WebView | data never leaves machine | **PHI-safe default**; full 192³ teacher on user's GPU/MPS/CPU |
| **RemoteGPU** | HTTPS to Modal / vast / JS2 / cloud | leaves machine — needs BAA/consent | no-GPU users, batch TotalSegmentator, biggest models |

Platform transport nuances (established): loopback works on desktop (Chrome/Firefox/Edge; Safari has a
WebKit mixed-content quirk — serve viewer from `localhost` too) and Android (foreground service, port
>1024); **iOS needs the helper to *be* a WebView-hosting app** (app-foreground = server-up), not a
background daemon. Chrome's **Local Network Access** prompt (~Chrome 142) means expect a one-time user
grant on first loopback connect.

### 2.3 The broker — two checks

**A. One-time device probe** (cached per browser/GPU): request raised fp16 WebGPU limits; if granted,
run a reference patch forward and time it. Classifies the device: `local-inference: fast / slow /
unavailable`, and selects the **largest student variant** that hits interactive latency here.

**B. Per-dataset fit analysis:** given volume dims/spacing + the target structure, compute and display,
live, as the user drags crop/resample:
- does the resampled whole volume fit the **render budget**;
- at what spacing the target fits one patch (FOV = patch × spacing);
- **AutoZoom pass-count** = `ceil(target_extent / (patch × spacing))` ⇒ predicted latency;
- the honest three-column comparison (this browser / helper / server) in real numbers.

Example readout the broker renders:

> **This browser (discrete GPU, 8 GB):** local nnInteractive ✓. At 0.6 mm your ROI fits one patch ⇒
> **~1.4 s/click, no upload, no PHI leaves this machine.** At 0.3 mm ⇒ AutoZoom (3 passes, ~4 s) or crop.
> **Helper app:** any spacing, full volume, ~0.4 s/click, still local/PHI-safe.
> **Server (your key):** same, plus batch TotalSegmentator.

### 2.4 Model as a routed resource — provenance + promote

The [capability manifest](CAPABILITY-MANIFEST.md) adds a **model-variant axis**. Because the browser
may use a 96³ student while the server uses 192³ L, identical prompts can yield slightly different masks.
Therefore:
- **Every segmentation is tagged with the variant that produced it** (provenance).
- **"Finalize on the full model"**: re-run the *accepted prompt sequence* (the interaction log we
  already keep) through the 192³ teacher on helper/server to produce the canonical result. Interact
  cheaply/locally; promote to full fidelity on demand — a natural `role=module` write-back under
  SlicerLive's authority model.
- A user-facing **quality/latency slider** (96³ ↔ 192³) with the same transparency: live clicks/sec +
  "lite model may need extra clicks and differs from canonical until promoted."

---

## 3. The student model family

| Variant | Patch | Backbone | Target tier | Primary domains |
|---|---|---|---|---|
| `nnlive-96` | 96³ | ResEnc-M/S | phones, iGPU, stock WebGPU | US, echo, small MR structures |
| `nnlive-128` | 128³ | ResEnc-M | mainstream laptops | MR, small-to-mid structures |
| `nnlive-160` | 160³ | ResEnc-L/M | discrete GPU / Apple Silicon | near-teacher quality |
| `teacher-192` | 192³ | ResEnc-L | helper / server only | canonical / promote target |

Additional knobs (each stated as a named, validated tradeoff): network width (L→M/S), AutoZoom cap,
input-channel subset (drop unsupported prompt types), int8 weight quantization (validate — WebGPU fp16
overflow is a known risk).

---

## 4. The training pipeline (distillation)

- **Self-labeling:** sample public volumes → simulate interaction sequences (reuse nnInteractive's
  interaction-simulation code) → teacher forward = soft-label target. **No ground-truth labels needed.**
- **Loss:** Dice + KL on logits (student vs teacher), student at the smaller patch/plans.
- **Setup:** teacher (~4 GB) + student (<1 GB) co-resident on one GPU; or precompute a teacher-label
  cache to local NVMe (removes live-teacher bottleneck, essential for multi-GPU feeding).
- **Config:** produce per-variant nnU-Net **plans files** (patch, spacing, arch); nnU-Net's
  ExperimentPlanner already supports budget-driven patch sizing.
- **Automatic eval (the free payoff):** held-out (image, prompt-sequence) → run teacher **and** student
  → **student-vs-teacher Dice / surface distance**, sliced by structure × modality × patch. Plus
  **Dice-vs-clicks curves** per variant. A small human-labeled set confirms the teacher is good on
  target domains; the relative student↔teacher curve — what the quality slider needs — is free.

---

## 5. Phased roadmap

Each milestone has a **gate**: a measured result that must hold before scaling to the next.

### M0 — Export & operator-coverage spike *(1–2 wks dev + a few A100-hrs)*  ⟵ **start here, derisks everything**
- Export the stock **192³** (and one **128³** config) nnInteractive graph to ONNX.
- Load in **ORT-Web WebGPU** in Chrome; confirm **every op stays on WebGPU** (no CPU fallback).
- Measure real per-patch latency on 2–3 reference GPUs (discrete, Apple Silicon, iGPU).
- **Gate:** a 128³ (or smaller) student-shaped graph runs fully on WebGPU at interactive latency. If a
  core 3D op won't run, decide: swap op / custom kernel / choose a WebGPU-friendly student arch.

### M1 — Distillation pipeline + first variant *(2 wks dev + 1–3 A100-days; ~$30–110 GCP Spot)*
- Adapt nnInteractive's trainer: distillation loss, smaller-patch plans, interaction simulation, spot
  **checkpoint/auto-resume**.
- Train **one `nnlive-128`** scoped to one modality (e.g., prostate/lymph-node MR).
- Build the **automatic eval** harness (student-vs-teacher Dice + clicks curves).
- **Gate:** `nnlive-128` within an agreed Dice-vs-teacher band on target structures; ONNX-exports;
  runs on WebGPU (M0 path).

### M2 — LiveModule browser client (InBrowser backend) *(3 wks dev)*
- Port the interaction-session orchestration to TS (±channel encoding, AutoZoom crop, target-buffer
  accumulation — mirrors the existing server endpoints).
- Implement the **inference API** (§2.1) over ORT-Web; wire results to a `SegmentationDM` in SlicerLive.
- **Device probe** (§2.3-A): raised-limits request + reference-forward timing + variant selection.
- **Gate:** end-to-end point/scribble segmentation of a small US/MR volume, in-browser, no server.

### M3 — Broker + fit analysis + crop/resample UI *(2–3 wks dev)*
- **Per-dataset fit analysis** (§2.3-B): FOV/AutoZoom/render-budget math + live readout.
- Crop/resample controls with real per-device numbers; the three-column comparison.
- **Capability manifest** consumption + variant fetch from a **CORS Ceph bucket**.
- **Gate:** the broker correctly classifies a set of (device × dataset) cases and its predicted
  latencies match measured within tolerance.

### M4 — LocalHelper + RemoteGPU backends + provenance/promote *(3 wks dev)*
- **LocalHelper**: package the existing nnInteractive server as a one-click installer / Docker;
  loopback transport; iOS WebView-host variant; LNA permission handshake.
- **RemoteGPU**: reuse `server/modal_app.py`; key-based routing to user's Modal/vast/cloud endpoint.
- **Provenance tags** + **finalize-on-full-model** (replay interaction log through teacher).
- **Gate:** same LiveScene segmented across all three backends; promote reproduces canonical mask.

### M5 — Student family + published models *(2–4 days GPU on 4×H100, one variant/GPU; ~$80–320 Spot)*
- Train **`nnlive-96 / 128 / 160`** across target domains; run full automatic eval; publish weights +
  **manifest** to the CORS bucket with **CC-BY-NC-SA** labeling.
- **Gate:** each variant has published Dice-vs-teacher + clicks curves; broker selects correctly across
  the device tiers (phone → workstation).

### M6 — SlicerLive integration & polish *(ongoing)*
- Register nnLive as a `LiveModule` in SlicerLive; agent tool-schema (`runModule`) exposure;
  live.slicer.org wiring; docs + demos on IDC/public data.

---

## 6. Repository layout

```
nnLive/
├── docs/
│   ├── DEVELOPMENT_PLAN.md      # this file
│   └── CAPABILITY-MANIFEST.md   # model-variant + device-class manifest spec
├── training/                    # distillation trainer, plans, automatic eval  (Python)
├── livemodule/                  # inference API, broker, ORT-Web backend        (TypeScript)
├── server/                      # reference GPU server (nnInteractive Modal)    (Python)
└── manifest/                    # published capability manifest + weight entries (JSON)
```

---

## 7. Cross-repo dependencies

- **SlicerLive** (`~/slicer/SlicerLive`) — the LiveScene protocol, `SegmentationDM`, the viewer,
  content-addressed blob channel, authority/lease model. nnLive is a participant in it.
- **SlicerNNInteractive** (`~/slicer/latest/SlicerNNInteractive`) — the reference server + Modal
  deployment (`server/modal_app.py`), and the interaction-simulation / session logic we port and reuse.
- **nnInteractive / nnU-Net** (upstream) — teacher weights, trainer, ExperimentPlanner.
- **Hosting** — JS2 Ceph CORS bucket (weights + manifest), per SlicerLive's hosting pattern.

---

## 8. Risks & open questions

1. **Operator coverage (M0)** — the make-or-break gate. Mitigation: pick a WebGPU-friendly student arch
   if a stock op won't run; custom WGSL kernel as fallback.
2. **Accuracy floor of 96³** — may be too weak for some structures. Mitigation: publish honest per-class
   numbers; let the broker refuse local for structures where the lite model underperforms.
3. **Data-loading bottleneck at multi-GPU** — precompute teacher-label cache to NVMe; size vCPU/RAM.
4. **iOS reach** — background daemon impossible; the WebView-host app (App Store, ~$0.99) is the answer,
   but is a native build, not browser-only. Desktop/Android are the browser-only majority.
5. **Cross-variant reproducibility** — addressed by provenance + promote, but needs clear UX so users
   know when they're looking at a lite vs canonical result.
6. **Teacher licensing propagates** — students are NC-SA derivatives (see §10).

---

## 9. Immediate next actions

1. `git init` + push nnLive to GitHub (this scaffold). *(pending user go-ahead on remote)*
2. **M0 spike**: export nnInteractive to ONNX; test ORT-Web WebGPU operator coverage on one desktop +
   one Apple Silicon machine. This single result decides whether browser-only is viable at all.
3. Stand up the JS2 (or GCP Spot A100) training instance; port nnInteractive's trainer with a
   distillation loss; validate the self-labeling loop on a tiny run before the M1 variant.

---

## 10. Licensing

- **Repo code: Apache-2.0.**
- **Teacher weights: CC-BY-NC-SA 4.0** (nnInteractive). **Distilled student weights are derivatives →
  inherit non-commercial + share-alike.** Every published weight/manifest entry must carry the license
  and attribution. This is compatible with the SlicerLive research context but **precludes commercial
  redistribution** of the students without a separate arrangement with the upstream authors.
