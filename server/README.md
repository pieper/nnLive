# server/ — reference GPU server (RemoteGPU backend)

The `RemoteGPU` backend reuses the existing **nnInteractive Slicer server** and its **Modal.com**
deployment, which is already a drop-in HTTP endpoint (`/upload_image`, `/add_point_interaction`,
`/add_bbox_interaction`, `/add_lasso_interaction`, `/add_scribble_interaction`, `/reset`, `/health`).

Canonical source today: `~/slicer/latest/SlicerNNInteractive/server/` — notably `modal_app.py`
(nnInteractive v2 on an A10G, weights cached in a Modal Volume, warm-container session state). That
file is the template for nnLive's RemoteGPU endpoints and for the LocalHelper packaging.

## Role in nnLive

- **RemoteGPU**: the broker routes here when the user supplies a Modal/vast/cloud key (see
  `servers[]` in [`../docs/CAPABILITY-MANIFEST.md`](../docs/CAPABILITY-MANIFEST.md)).
- **LocalHelper**: the same server, packaged as a one-click installer / Docker bound to `localhost`
  — the PHI-safe default. iOS ships it as a WebView-hosting app instead of a background daemon.
- **Promote target**: hosts the canonical `teacher-192` model for "finalize on full model".

## TODO

- Vendor / submodule the reference server; add `/finalize` (replay an interaction log through the
  192³ teacher) and multi-session support (per-session state, not module-global).
- Package the LocalHelper installer (desktop + Android foreground service + iOS WebView app).
