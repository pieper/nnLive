# SAM-Med3D in the browser (WebGPU) — nnLive port

A WebGPU port of **SAM-Med3D-turbo** (Apache-2.0), the encode-once / decode-per-click 3D
promptable segmenter. This is the "smarter, faster GrowCut" path: the heavy image encoder runs
**once**, then every click runs only a tiny decoder — so refinement is genuinely interactive
instead of nnInteractive's 4–40 s full-forward per click.

## Try it
```
cd livemodule/probe
python3 serve.py 8799          # COOP/COEP on -> wasm threads; self-hosted ORT under ort/
# open http://localhost:8799/sammed3d.html  (Chrome/Edge, WebGPU)
```
- **Left-click** = positive, **Shift/right-click** = negative, scroll/slider = slice.
- First click drops a **128³ box** (~192 mm) and runs the encoder once (~1 s load + ~3 s encode
  on wasm). After that each click runs only the decoder on WebGPU (~0.1 s) and the mask
  accumulates, respecting earlier clicks (SAM-Med3D's own previous-mask input is fed back).
- Click outside the box to start a fresh region; **clear** resets.

## What runs where, and why
| stage | model | EP | cost | note |
|---|---|---|---|---|
| encode (once/box) | `encoder.onnx` (372 MB fp32) | **wasm** | ~3 s | ViT; wasm has full op coverage (Einsum/Erf/Range) and, crucially, **no single long GPU submission** so it can't trip the M4 WindowServer GPU-watchdog that crashed the teacher port |
| decode (per click) | `decoder.onnx` (32 MB fp32) | **WebGPU** (+wasm fallback) | ~0.1 s | prompt-encoder + mask-decoder, tiny |

**Verified in a real browser** (headless Chrome, Metal) against a correctly-loaded PyTorch
reference, on the real CT crop (not just the trace input):
- encoder-on-wasm embedding vs torch: **corr 1.000000, max|diff| 3e-6**
- decoder-on-webgpu low-res logits vs torch: **max|diff| 9e-6, 100 % mask-sign agreement**
- full in-browser pipeline (encode→click→trilinear-upsample→threshold): **898 mask voxels,
  byte-for-byte the same count PyTorch produces** for the same crop+click.

Self-tests: `sammed3d-selftest.html` (numeric, auto-posts PASS/FAIL) and `sammed3d.html?auto=1`
(full pipeline; clicks a real IDC-seg voxel, checks the mask centroid lands on the click).

> **Validation gotcha (learned the hard way):** the turbo checkpoint is wrapped in
> `{"model_state_dict": ...}`. `load_state_dict(sd, strict=False)` **silently loads nothing** if
> you forget to unwrap — you get a *randomly initialized* model that still runs and still
> "matches" any ONNX exported from that same random instance. Always unwrap, and always validate
> ONNX on inputs **different from the export trace input**. (`export_sammed3d.py` unwraps; the
> models above are correct.)

## How it was built (reproducible)
Scripts live in `~/sammed3d_work/` (venv: python3.13 + torch 2.13 CPU):
1. `export_sammed3d.py` — loads turbo ckpt (`vit_b_ori`: embed 768, 12 blocks, **128³ input →
   8³ embedding**, neck→384). Splits into **encoder** (image→`[1,384,8,8,8]`) and **decoder**
   (image_embeddings + point_coords + point_labels + prev_mask + has_mask → `[1,1,32,32,32]`
   logits). The prompt encoder is inlined and its boolean label-indexing rewritten as
   arithmetic (Equal→Cast→Mul→Add) so it exports cleanly. `image_pe` baked as a constant.
   Legacy TorchScript exporter (`dynamo=False`).
2. `decoder_surgery.py` — the only WebGPU-unsupported op is `ConvTranspose3d` (×2, k2/s2 in the
   mask upscaler). Rewritten to the proven **1×1×1 Conv → Reshape → 8D-Transpose → Reshape**
   pixel-shuffle (math-identical). Emits `decoder_webgpu_fp32.onnx` (used) + an fp16 (currently
   invalid — a `convert_float_to_float16` Cast bug; not needed, decoder is small).
3. `validate_and_ops.py` — numeric check vs torch + op histograms.

Preprocessing replicated from SAM-Med3D's own inference (`utils/infer_utils.py`): resample to
**1.5 mm** iso, 128³ crop, **Z-normalize over voxels with HU>0**. Points in (d,h,w)=(z,y,x)
crop-voxel order. A padding point (label −1) is appended per SAM-Med3D convention.

## Open design questions (asked, worth pursuing)

### Native-resolution mode
The ViT is architecturally locked to **128³ input → 8³ tokens** (patch-embed stride-16, baked
8³ `pos_embed`, window/rel-pos tables sized to the grid). So "native res" is not a free switch.
Options, easiest first:
- **Spacing knob (recommended):** the 128³ box can be sampled at *any* voxel spacing. Sample it
  at the CT's native spacing instead of 1.5 mm → full detail over a smaller physical box (model
  still sees 128³). Needs the native-res source; the demo currently loads only the 1.5 mm cache.
- **Larger token grid (e.g. 256³→16³):** interpolate `pos_embed` (8³→16³) and the rel-pos
  tables; `get_rel_pos` already interpolates. It's an **untrained regime** (accuracy risk) and
  ~8× encoder cost — prototype-worthy, not a sure win.
- **Tile & stitch** the 128³ model at native spacing over a big organ. More engineering; exact.

### Cohort pre-encode (LNQ2023 etc.)
The image embedding is just **`[1,384,8,8,8]` = 786 KB fp32 (≈393 KB fp16)**. So you can run the
encoder **offline on GPU over a whole cohort**, cache the tiny embeddings, and ship only those
+ the 32 MB decoder. The browser then **skips encoding entirely** and clicks are ~0.1 s from the
first interaction. The worker already caches the embedding; the drop-in is a `loadEmbedding`
message that accepts a precomputed `[1,384,8,8,8]` blob instead of running `encode`. This is the
natural way to "jump straight into segmenting abdomen/pelvis nodes" on a target dataset.

### Patient-space resampling (required for real integration)
This test demo shortcuts geometry by using an **already-1.5 mm-iso-resampled** volume and cropping
in voxel space. The real LiveModule must work in **patient (RAS) space**: define the 128³ box as a
physical region, resample the *native* image into it via its IJK→RAS affine (handling anisotropic
spacing, direction cosines, oblique/gantry-tilt acquisitions), place click points in patient
coordinates, and map the output mask back through the inverse affine onto the native grid. This is
exactly SAM-Med3D's own `tio.Resample`(1.5 mm)+`ToCanonical` pipeline — it must be carried into the
browser (ship the affine alongside the voxels) so different scan geometries segment correctly.

## License
SAM-Med3D weights & code: **Apache-2.0** (commercial-OK, no NC clause — unlike nnInteractive
CC-BY-NC-SA or VISTA3D). This port inherits Apache-2.0.
