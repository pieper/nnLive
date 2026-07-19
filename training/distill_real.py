"""Real-data distillation: student learns teacher masks on real IDC CT/MR patches.

IDC segs are used ONLY to place realistic prompts (fg/bg clicks + a scribble) on real anatomy; the
TEACHER produces the mask the student learns (self-labeling). Consumes the cache the filler produces,
keeps decoded cases in a RAM pool, samples MANY patches per case, and rescans for new cases while
training (download overlaps compute). Early-stops on a plateau of good student-vs-teacher Dice; hard
8h cap. Reports the new student vs the synthetic-data student on a held-out real eval set.
"""
import os, time, glob, threading, argparse
import numpy as np
import torch, torch.nn.functional as F
from student import Student

WEIGHTS = os.path.expanduser("~/.nninteractive_weights/nnInteractive_v1.0")
CACHE = os.path.expanduser("~/idc_cache")
SYNTH = os.path.expanduser("~/nnlive_student/student.pt")


# ---------- case pool (RAM cache of decoded npz, rescanned while training) ----------
class CasePool:
    def __init__(self, hold_out=16, max_cases=80):
        self.max_cases = max_cases
        self.cases = {}          # name -> (vol f32, label u8)
        self.order = []
        self.eval_names = set()
        self.hold_out = hold_out
        self.lock = threading.Lock()

    def _load(self, path):
        d = np.load(path)
        return d["vol"].astype(np.float32), d["lab"].astype(np.uint8)

    def rescan(self):
        files = sorted(glob.glob(os.path.join(CACHE, "*.npz")))
        for p in files:
            name = os.path.basename(p)
            if name in self.cases:
                continue
            try:
                v, l = self._load(p)
            except Exception:
                continue
            with self.lock:
                # first `hold_out` cases are the frozen eval set
                if len(self.eval_names) < self.hold_out and name not in self.eval_names:
                    self.eval_names.add(name)
                self.cases[name] = (v, l)
                self.order.append(name)
                # evict oldest TRAIN case if over budget (never evict eval cases)
                while len(self.order) > self.max_cases:
                    old = self.order.pop(0)
                    if old in self.eval_names:
                        self.order.append(old); break
                    self.cases.pop(old, None)
        return len(self.cases)

    def train_names(self):
        with self.lock:
            return [n for n in self.order if n not in self.eval_names]


# ---------- patch + prompt synthesis ----------
def _ball(r):
    rr = int(np.ceil(r)); zz, yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1, -rr:rr + 1]
    m = (zz * zz + yy * yy + xx * xx) <= r * r
    return np.stack([zz[m], yy[m], xx[m]], 1)


BALL3, BALL1 = _ball(3), _ball(1)


def _paint(ch, c, ball):
    p = ball + c; ps = ch.shape[0]
    ok = ((p >= 0) & (p < ps)).all(1); p = p[ok]
    ch[p[:, 0], p[:, 1], p[:, 2]] = 1.0


def _crop(vol, label, center, ps):
    lo = [int(center[i] - ps // 2) for i in range(3)]
    v = np.full((ps, ps, ps), float(vol.min()), np.float32); l = np.zeros((ps, ps, ps), np.uint8)
    sl = [max(0, lo[i]) for i in range(3)]; sh = [min(vol.shape[i], lo[i] + ps) for i in range(3)]
    dl = [sl[i] - lo[i] for i in range(3)]
    ss = tuple(slice(sl[i], sh[i]) for i in range(3))
    ds = tuple(slice(dl[i], dl[i] + sh[i] - sl[i]) for i in range(3))
    v[ds] = vol[ss]; l[ds] = label[ss]
    return v, l


def make_input(vol, label, ps, rng):
    """Build one 8-channel input from a real case + seg-derived prompts, or None if unusable."""
    segs = [int(s) for s in np.unique(label) if s > 0]
    if not segs:
        return None
    segid = int(rng.choice(segs))
    fg_all = np.argwhere(label == segid)
    if len(fg_all) < 20:
        return None
    center = fg_all[rng.integers(len(fg_all))] + rng.integers(-ps // 4, ps // 4 + 1, size=3)
    v, l = _crop(vol, label, center, ps)
    fgm = l == segid
    if fgm.sum() < 10:
        return None
    x = np.zeros((8, ps, ps, ps), np.float32)
    x[0] = (v - v.mean()) / (v.std() + 1e-5)                          # ch0 image (z-score)
    fgv = np.argwhere(fgm)
    for c in fgv[rng.choice(len(fgv), min(3, len(fgv)), replace=False)]:
        _paint(x[4], c, BALL3)                                        # ch4 points_pos
    bgm = (~fgm) & (l > 0)
    bgv = np.argwhere(bgm) if bgm.sum() > 10 else np.argwhere((~fgm) & (x[0] > np.percentile(x[0], 60)))
    if len(bgv):
        for c in bgv[rng.choice(len(bgv), min(3, len(bgv)), replace=False)]:
            _paint(x[5], c, BALL3)                                    # ch5 points_neg
    if len(fgv) >= 2:                                                 # ch6 scribble_pos
        a, b = fgv[rng.integers(len(fgv))], fgv[rng.integers(len(fgv))]
        for t in np.linspace(0, 1, 12):
            _paint(x[6], (a + (b - a) * t).astype(int), BALL1)
    return x


def batch(pool, bs, ps, rng, dev):
    xs = []
    while len(xs) < bs:
        names = pool.train_names()
        if not names:
            pool.rescan(); time.sleep(1); continue   # discover new cases while stalled (never deadlock)
        v, l = pool.cases[names[rng.integers(len(names))]]
        xi = make_input(v, l, ps, rng)
        if xi is not None:
            xs.append(xi)
    return torch.from_numpy(np.stack(xs)).to(dev)


# ---------- losses / eval ----------
def load_teacher(dev):
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    s = nnInteractiveInferenceSession(device=torch.device(dev), use_torch_compile=False,
                                      verbose=False, do_autozoom=True, interactions_storage="auto")
    s.initialize_from_trained_model_folder(WEIGHTS)
    net = getattr(s.network, "_orig_mod", s.network)
    if hasattr(net, "decoder") and hasattr(net.decoder, "deep_supervision"):
        net.decoder.deep_supervision = False
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def kd_loss(s_logits, t_logits):
    t_soft = F.softmax(t_logits, 1)
    kl = F.kl_div(F.log_softmax(s_logits, 1), t_soft, reduction="none").sum(1).mean()
    t_hard = t_logits.argmax(1).float()
    s_fg = F.softmax(s_logits, 1)[:, 1]
    inter = (s_fg * t_hard).sum((1, 2, 3))
    dice = 1 - ((2 * inter + 1) / (s_fg.sum((1, 2, 3)) + t_hard.sum((1, 2, 3)) + 1)).mean()
    return kl + dice, kl.item(), dice.item(), t_hard.mean().item()


@torch.no_grad()
def agree_dice(student, teacher, evalset):
    """Mean student-vs-teacher fg Dice over the fixed eval patches."""
    ds = []
    for x in evalset:
        with torch.autocast("cuda", dtype=torch.float16):
            t = teacher(x).argmax(1)
            s = student(x).argmax(1)
        tf, sf = (t == 1).float(), (s == 1).float()
        inter = (tf * sf).sum()
        ds.append(((2 * inter + 1) / (tf.sum() + sf.sum() + 1)).item())
    return float(np.mean(ds)) if ds else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--max_hours", type=float, default=8.0)
    ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--plateau_minutes", type=float, default=30.0,   # stop only after this long with NO new best
                    help="early-stop if the best eval Dice hasn't improved for this many minutes")
    ap.add_argument("--snapshot_every", type=int, default=0,         # 0 = OFF (no intermediate capture)
                    help="every N steps, save a lightweight fp16 checkpoint to OUT/snapshots for the "
                         "learning-curve animation. A separate publish_snapshots.py exports+uploads them. "
                         "Leave 0 for browser-student runs; enable for the future size-sweep experiments.")
    ap.add_argument("--out", default=os.path.expanduser("~/nnlive_student_real"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda:0"
    rng = np.random.default_rng(0)

    pool = CasePool()
    need = pool.hold_out + 4                        # full eval set + a few train cases before starting
    while pool.rescan() < need:
        print(f"waiting for cache… {len(pool.cases)}/{need}", flush=True); time.sleep(15)

    teacher = load_teacher(dev)
    student = Student(cin=8, cout=2, base=args.base).to(dev)
    synth = Student(cin=8, cout=2, base=args.base).to(dev)
    synth.load_state_dict(torch.load(SYNTH, map_location=dev)); synth.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    # frozen eval set: fixed patches from held-out cases
    ev_rng = np.random.default_rng(123)
    evalset = []
    for name in list(pool.eval_names):
        v, l = pool.cases[name]
        xi = make_input(v, l, args.patch, ev_rng)
        if xi is not None:
            evalset.append(torch.from_numpy(xi[None]).to(dev))
    synth_dice = agree_dice(synth, teacher, evalset)
    print(f"EVALSET {len(evalset)} patches | SYNTHETIC-student vs teacher Dice = {synth_dice:.3f}", flush=True)

    t0 = time.time(); tw = t0
    best = 0.0
    last_improve = t0                              # wall-clock of the last new best (drives plateau stop)
    step = 0
    while True:
        step += 1
        if step % 200 == 0:
            pool.rescan()                          # pick up newly downloaded cases
        x = batch(pool, args.bs, args.patch, rng, dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            t_logits = teacher(x)
        with torch.autocast("cuda", dtype=torch.float16):
            s_logits = student(x)
            loss, kl, dice, tfg = kd_loss(s_logits.float(), t_logits.float())
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

        if step % 20 == 0:
            dt = (time.time() - tw) / 20; tw = time.time()
            print(f"step {step} loss {loss.item():.3f} kl {kl:.3f} dice {dice:.3f} tfg {tfg*100:.1f}% "
                  f"cases {len(pool.cases)} {dt*1000:.0f}ms/it t {(time.time()-t0)/3600:.2f}h", flush=True)

        if args.snapshot_every and step % args.snapshot_every == 0:
            # cheap: just dump fp16 weights to disk (~47MB). publish_snapshots.py (separate process)
            # exports each to browser-native ONNX and uploads it, so we never export on the training GPU.
            snapdir = os.path.join(args.out, "snapshots"); os.makedirs(snapdir, exist_ok=True)
            sd = {k: v.detach().half().cpu() for k, v in student.state_dict().items()}
            tmp = os.path.join(snapdir, f".step_{step:06d}.pt")   # write-then-rename so the publisher never sees a partial
            torch.save(sd, tmp); os.replace(tmp, os.path.join(snapdir, f"step_{step:06d}.pt"))

        if step % args.eval_every == 0 and evalset:
            student.eval(); d = agree_dice(student, teacher, evalset); student.train()
            if d > best + 0.002:                    # new best -> keep it and reset the plateau clock
                best = d; last_improve = time.time()
                torch.save(student.state_dict(), os.path.join(args.out, "student.pt"))
            flat_min = (time.time() - last_improve) / 60
            print(f"EVAL step {step} real-student-vs-teacher Dice {d:.3f} (best {best:.3f}, "
                  f"synth {synth_dice:.3f}, flat {flat_min:.1f}min)", flush=True)
            elapsed_h = (time.time() - t0) / 3600
            # train for the full budget for max Dice; stop early only on a real plateau (no new best for N min)
            if flat_min >= args.plateau_minutes or elapsed_h >= args.max_hours:
                reason = "plateau" if flat_min >= args.plateau_minutes else "time_cap"
                print(f"STOP ({reason}) best {best:.3f} synth {synth_dice:.3f} at {elapsed_h:.2f}h "
                      f"(flat {flat_min:.1f}min)", flush=True)
                break

    # final save + browser-native ONNX export
    torch.save(student.state_dict(), os.path.join(args.out, "student_final.pt"))
    with open(os.path.join(args.out, "DONE"), "w") as f:
        f.write(f"best_dice={best:.3f} synth_dice={synth_dice:.3f} hours={(time.time()-t0)/3600:.2f}\n")
    print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
