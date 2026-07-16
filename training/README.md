# training/ — student distillation

Distills the full 192³ nnInteractive **teacher** into smaller-patch **student** variants
(`nnlive-96 / 128 / 160`) that fit the browser's WebGPU budget. See
[`../docs/DEVELOPMENT_PLAN.md`](../docs/DEVELOPMENT_PLAN.md) §3–§4.

## Approach

- **Self-labeling:** the teacher generates unlimited `(image, prompt-sequence, mask)` targets — no
  ground-truth labels required. Reuses nnInteractive's interaction-simulation code.
- **Loss:** Dice + KL(logits) of student vs teacher, student at a smaller-patch nnU-Net plan.
- **Setup:** teacher (~4 GB) + student (<1 GB) co-resident on one GPU, or precomputed teacher-label
  cache on local NVMe for multi-GPU feeding.
- **Spot-safe:** checkpoint every epoch + auto-resume (GCP Spot / preemptible; nnU-Net checkpoints already).

## Planned layout

```
training/
├── distill.py            # trainer: distillation loss + interaction simulation + spot-resume
├── plans/                # per-variant nnU-Net plans (patch/spacing/arch)  e.g. nnlive-128.json
├── cache/                # teacher-label cache builder (optional NVMe precompute)
├── export/               # PyTorch -> ONNX export + ORT-Web operator-coverage check (M0)
└── eval/                 # automatic student-vs-teacher Dice + Dice-vs-clicks curves
```

## Compute (from the plan)

Narrow variant ~1–3 A100-days (~$30–110 GCP Spot); family of 3 ~$80–320 Spot, or ~2–4 days on 4×H100
(one variant per GPU). JS2 A100/H100 is $0 (research allocation). A100 Spot is the cost sweet spot.

## First runs

1. **M0 export spike** (`export/`) — the make-or-break gate: ONNX export + ORT-Web WebGPU coverage.
2. **M1** — one `nnlive-128` on one modality + the automatic eval harness.
