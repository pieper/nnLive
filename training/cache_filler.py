"""Background producer: download IDC CT/MR cases, resample to 1.5mm, write atomic .npz to the cache.

Runs concurrently with training (which consumes the cache). Atomic tmp+rename so the trainer never
reads a partial file. int16 image + uint8 labelmap keeps the cache small.
"""
import json, os, tempfile, random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import idc_data

OUT = os.path.expanduser("~/idc_cache")
TMPDIR = os.path.join(OUT, ".tmp")           # trainer globs OUT/*.npz, never this subdir
os.makedirs(TMPDIR, exist_ok=True)
TARGET = int(os.environ.get("TARGET", "400"))
SPACING = 1.5

rows = json.load(open(os.path.expanduser("~/segroulette.json")))["rows"]
ctmr = [r for r in rows if r.get("m") in ("CT", "MR")]
random.seed(0); random.shuffle(ctmr)


def one(r):
    name = r["s"][:16]
    dst = os.path.join(OUT, name + ".npz")
    if os.path.exists(dst):
        return "skip"
    try:
        with tempfile.TemporaryDirectory() as wd:
            v, l = idc_data.load_case(r, wd, spacing=SPACING)
        if int((l > 0).sum()) < 200 or min(v.shape) < 16:
            return "tiny"
        tmp = os.path.join(TMPDIR, name + ".npz")   # savez keeps this name (already .npz)
        np.savez_compressed(tmp, vol=v.astype(np.int16), lab=l.astype(np.uint8),
                            m=r["m"], col=r["col"])
        os.replace(tmp, dst)                    # atomic publish into OUT/name.npz
        return "ok"
    except Exception as e:
        return "fail:" + str(e)[:60]


done = len([f for f in os.listdir(OUT) if f.endswith(".npz")])
print(f"start: {done} cached, target {TARGET}", flush=True)
with ThreadPoolExecutor(max_workers=6) as ex:
    for i, res in enumerate(ex.map(one, ctmr[: TARGET * 3])):
        if res == "ok":
            done += 1
        if res.startswith("fail") or i % 10 == 0:
            print(f"[{done}/{TARGET}] case{i} {res}", flush=True)
        if done >= TARGET:
            break
print("FILLER_DONE", done, flush=True)
