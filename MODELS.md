# Model weights & license

The nnLive student model is **distilled from the [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) teacher weights**, which are released under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**. As a derivative, **the nnLive weights inherit those terms:**

- **BY** — attribution to nnInteractive (Isensee, Rokuss, et al., MIC-DKFZ) is required.
- **NC** — **non-commercial use only.**
- **SA** — redistribution of the weights (or derivatives) must be under the same CC BY-NC-SA 4.0 license.

This applies **only to the model weights**. The nnLive source code in this repository is separately licensed under **Apache-2.0** (see [`LICENSE`](LICENSE)).

## Weight artifacts (hosting)

The large weight file is **not committed to the repository**. It is served from a public, **CORS-enabled** JS2 (Jetstream2 / NSF ACCESS) Ceph object bucket so the browser demo can fetch it cross-origin:

| File | URL |
|---|---|
| `perclick_192.weights.bin` (fp16 decode net, 188 MB) | `https://js2.jetstream-cloud.org:8001/swift/v1/nnlive-models/perclick_192.weights.bin` |

The small graph descriptors and the image-trunk weights (`docs/models/pathA/faithful/trunk8_192.*`, `perclick_192.graph.json`) are in the repository. The demo fetches the decode weights from the bucket URL above; `?weights=<url>` overrides it for local development.

> Note: GitHub **release assets are not usable** here — `release-assets.githubusercontent.com` does not send `Access-Control-Allow-Origin`, so a cross-origin browser `fetch()` of a release asset is blocked. The CORS-enabled bucket avoids both that and the 100 MB per-file repo limit.

## Provenance

- **Teacher:** nnInteractive v1.0 (`Dataset225_nnInteractiveV2`, ResEnc-L, patch 192³).
- **Distillation:** teacher-in-the-loop knowledge distillation with a faithful autoregressive interaction rollout (global z-score, EDT r4 point prompts, decay 0.9, `prev_seg` feedback, zoom augmentation). Agreement with the teacher ≈ 0.74 Dice on held-out refinement rollouts.
