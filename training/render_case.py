"""Render one IDC case's most-segmented slice with the SEG overlaid, to visually verify the pull."""
import json, os, sys, tempfile, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import idc_data

rows = json.load(open(os.path.expanduser("~/segroulette.json")))["rows"]
COLL = sys.argv[1] if len(sys.argv) > 1 else "c4kc_kits"     # collection substring
IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/idc_slice.png")
cases = [r for r in rows if r.get("m") in ("CT", "MR") and COLL in r["col"]]
r = cases[IDX % len(cases)]
print("case:", r["m"], r["col"], r.get("sd"), flush=True)

with tempfile.TemporaryDirectory() as wd:
    vol, lab = idc_data.load_case(r, wd)

z = int((lab > 0).sum(axis=(1, 2)).argmax())
img, seg = vol[z], lab[z]
lo, hi = np.percentile(img, [1, 99])
imgn = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

plt.figure(figsize=(6, 6))
plt.imshow(imgn, cmap="gray")
plt.imshow(np.ma.masked_where(seg == 0, seg), cmap="autumn", alpha=0.4, interpolation="nearest")
segs = sorted(set(int(x) for x in np.unique(lab)) - {0})
plt.title(f"{r['m']} · {r['col']} · z={z}/{vol.shape[0]} · segs={segs}", fontsize=9)
plt.axis("off"); plt.tight_layout()
plt.savefig(OUT, dpi=100, bbox_inches="tight")
print(f"VOL {vol.shape} LABEL segs={segs} fg={int((lab>0).sum())} slice z={z}", flush=True)
