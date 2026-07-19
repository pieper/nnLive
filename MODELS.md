# Model weights & license

The nnLive student model is **distilled from the [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) teacher weights**, which are released under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**. As a derivative, **the nnLive weights inherit those terms:**

- **BY** — attribution to nnInteractive (Isensee, Rokuss, et al., MIC-DKFZ) is required.
- **NC** — **non-commercial use only.**
- **SA** — redistribution of the weights (or derivatives) must be under the same CC BY-NC-SA 4.0 license.

This applies **only to the model weights**. The nnLive source code in this repository is separately licensed under **Apache-2.0** (see [`LICENSE`](LICENSE)).

## Weight artifacts (GitHub release assets)

The large weight files are distributed as **GitHub release assets**, not committed to the repository:

| File | Contents |
|---|---|
| `perclick_192.weights.bin` | fp16 weights for the prompt-conditioned decode network (192³) |

The small graph descriptors and the image-trunk weights (`docs/models/pathA/faithful/trunk8_192.*`, `perclick_192.graph.json`) are in the repository. The live demo fetches `perclick_192.weights.bin` from the latest release; for local development it uses a local copy (`?weights=` overrides the URL).

## Provenance

- **Teacher:** nnInteractive v1.0 (`Dataset225_nnInteractiveV2`, ResEnc-L, patch 192³).
- **Distillation:** teacher-in-the-loop knowledge distillation with a faithful autoregressive interaction rollout (global z-score, EDT r4 point prompts, decay 0.9, `prev_seg` feedback, zoom augmentation). Agreement with the teacher ≈ 0.74 Dice on held-out refinement rollouts.
