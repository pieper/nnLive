"""Publish training snapshots as browser-native ONNX to the CORS Ceph bucket, for learning-curve animations.

Decoupled from training (run as a SEPARATE process, like cache_filler): watches OUT/snapshots/*.pt written
by distill_real.py --snapshot_every, exports each to fp16 browser-native ONNX, and uploads it to
    nnlive-models/anim/<tag>/step_XXXXXX.onnx
It also maintains  nnlive-models/anim/<tag>/frames.json  so a web player can enumerate the frames and
"scrub" the model's learning as an animation (load each ONNX, run on a fixed input, render the mask).

Exporting happens HERE, not in the training loop, so it never competes for the training GPU — CPU export
of the 23.5M-param student is quick. Idempotent: already-published steps are skipped, so it is safe to
run concurrently with training or as a one-shot post-pass (--once).

Usage (future size-sweep runs):
    python publish_snapshots.py --tag real_128_v1 --patch 128 --base 32           # watch + publish live
    python publish_snapshots.py --tag real_128_v1 --once                          # batch existing snapshots
"""
import os, glob, time, json, argparse, tempfile, subprocess
import torch, onnx
from onnxconverter_common import float16
from student import Student

CLOUD = "MED250016_IU"
SWIFT = "https://js2.jetstream-cloud.org:8001/swift/v1"
CONTAINER = "nnlive-models"


def token():
    return subprocess.check_output(
        ["openstack", f"--os-cloud={CLOUD}", "token", "issue", "-c", "id", "-f", "value"],
        text=True).strip()


def put_bytes(obj_name, data, ctype):
    """PUT raw bytes to the bucket (stdin, no temp file). Retries with a fresh token."""
    for _ in range(3):
        r = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-X", "PUT",
             "-H", f"X-Auth-Token: {token()}", "-H", f"Content-Type: {ctype}",
             "--data-binary", "@-", f"{SWIFT}/{CONTAINER}/{obj_name}"],
            input=data, stdout=subprocess.PIPE)
        if r.stdout.decode().strip() in ("201", "202"):
            return True
        time.sleep(3)
    return False


def export_fp16_onnx(pt_path, patch, base):
    """Load an fp16 snapshot .pt into the Student and export a browser-native fp16 ONNX; return its bytes."""
    m = Student(cin=8, cout=2, base=base)
    m.load_state_dict(torch.load(pt_path, map_location="cpu"))   # copy_ casts fp16 weights into the fp32 model
    m.eval()
    dummy = torch.zeros(1, 8, patch, patch, patch)
    with tempfile.TemporaryDirectory() as d:
        fp32 = os.path.join(d, "m.onnx")
        with torch.no_grad():
            torch.onnx.export(m, dummy, fp32, input_names=["input"], output_names=["logits"],
                              opset_version=18, do_constant_folding=True)
        mo = float16.convert_float_to_float16(onnx.load(fp32), keep_io_types=True)
        out = os.path.join(d, "m16.onnx")
        onnx.save_model(mo, out, save_as_external_data=False)
        with open(out, "rb") as f:
            return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/nnlive_student_real"))
    ap.add_argument("--tag", required=True, help="animation set name, e.g. real_128_v1")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--once", action="store_true", help="publish existing snapshots and exit (default: watch)")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()
    snapdir = os.path.join(args.out, "snapshots")

    done, frames = set(), []
    while True:
        pts = sorted(glob.glob(os.path.join(snapdir, "step_*.pt")))
        for p in pts:
            name = os.path.basename(p)
            if name in done:
                continue
            step = name[len("step_"):-3]
            obj = f"anim/{args.tag}/step_{step}.onnx"
            try:
                data = export_fp16_onnx(p, args.patch, args.base)
            except Exception as e:
                print(f"export {name} FAILED: {e}", flush=True)
                continue
            if put_bytes(obj, data, "application/octet-stream"):
                done.add(name); frames.append(f"step_{step}.onnx")
                manifest = json.dumps({"tag": args.tag, "patch": args.patch, "base": args.base,
                                       "frames": frames}, indent=2).encode()
                put_bytes(f"anim/{args.tag}/frames.json", manifest, "application/json")
                print(f"published {obj} ({len(data)//1024}KB) [{len(frames)} frames]", flush=True)
            else:
                print(f"upload {obj} FAILED (retry next poll)", flush=True)

        if args.once:
            break
        if os.path.exists(os.path.join(args.out, "DONE")) and all(os.path.basename(p) in done for p in pts):
            print("DONE sentinel seen and all snapshots published — exiting", flush=True)
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
