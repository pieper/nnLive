"""M1 distillation: teach the browser-native Student to imitate the nnInteractive teacher.

Self-labeling: the frozen teacher provides soft targets on synthetic promptable volumes (smooth
random image + random blobs; a positive-point prompt on one blob, encoded in the teacher's real
8-channel layout: ch0=image, ch4=points_pos). No ground truth needed. Teacher and student see the
SAME input at the SAME patch size (teacher is fully convolutional, so 128^3 is a valid evaluation).

NOTE (data caveat): synthetic blobs are a placeholder distribution to stand up the pipeline, get a
real time estimate, and produce a browser-native ONNX for latency testing. A production student
needs real volumes + nnInteractive's interaction simulation — that's the next data iteration. The
*latency* and *WebGPU-coverage* of the exported student are valid regardless of data quality.
"""
import os, time, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from student import Student

WEIGHTS = os.path.expanduser("~/.nninteractive_weights/nnInteractive_v1.0")
_grid = {}


def coord_grid(ps, device):
    if ps not in _grid:
        a = torch.arange(ps, device=device, dtype=torch.float32)
        _grid[ps] = torch.meshgrid(a, a, a, indexing="ij")
    return _grid[ps]


def make_batch(bs, ps, device):
    zz, yy, xx = coord_grid(ps, device)
    x = torch.zeros(bs, 8, ps, ps, ps, device=device)
    for b in range(bs):
        low = torch.randn(1, 1, max(2, ps // 16), max(2, ps // 16), max(2, ps // 16), device=device)
        img = F.interpolate(low, size=(ps, ps, ps), mode="trilinear", align_corners=False)[0, 0]
        nb = int(torch.randint(1, 4, (1,)).item())
        tgt = int(torch.randint(0, nb, (1,)).item())
        center = None
        for k in range(nb):
            c = torch.rand(3, device=device) * (ps * 0.5) + ps * 0.25
            r = torch.rand(1, device=device).item() * (ps * 0.13) + ps * 0.05
            d2 = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2
            img = img + (d2 <= r * r).float() * (1.0 + torch.rand(1, device=device).item())
            if k == tgt:
                center = c
        x[b, 0] = img
        # positive point as a small sphere (radius ~3 vox) in points_pos channel (input ch4)
        d2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
        x[b, 4] = (d2 <= (ps * 0.025) ** 2).float()
    m = x[:, 0].mean(dim=(1, 2, 3), keepdim=True)
    s = x[:, 0].std(dim=(1, 2, 3), keepdim=True) + 1e-5
    x[:, 0] = (x[:, 0] - m) / s
    return x


def load_teacher(device):
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    s = nnInteractiveInferenceSession(device=torch.device(device), use_torch_compile=False,
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
    t_soft = F.softmax(t_logits, dim=1)
    # per-voxel KL: sum over classes, mean over batch+spatial (NOT batchmean, which divides by bs only)
    kl = F.kl_div(F.log_softmax(s_logits, dim=1), t_soft, reduction="none").sum(1).mean()
    t_hard = t_logits.argmax(1).float()
    s_fg = F.softmax(s_logits, dim=1)[:, 1]
    inter = (s_fg * t_hard).sum(dim=(1, 2, 3))
    dice = 1 - ((2 * inter + 1) / (s_fg.sum(dim=(1, 2, 3)) + t_hard.sum(dim=(1, 2, 3)) + 1)).mean()
    return kl + dice, kl.item(), dice.item(), t_hard.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--out", default=os.path.expanduser("~/nnlive_student"))
    ap.add_argument("--smoke", type=int, default=0, help="run N steps, no checkpoint (timing only)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda:0"

    teacher = load_teacher(dev)
    student = Student(cin=8, cout=2, base=args.base).to(dev)
    print(f"student params {sum(p.numel() for p in student.parameters())/1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    N = args.smoke or args.steps
    t0 = time.time(); tw = t0
    for step in range(1, N + 1):
        x = make_batch(args.bs, args.patch, dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            t_logits = teacher(x)
            if isinstance(t_logits, (list, tuple)):
                t_logits = t_logits[0]
        with torch.autocast("cuda", dtype=torch.float16):
            s_logits = student(x)
            loss, kl, dice, tfg = kd_loss(s_logits.float(), t_logits.float())
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step % 20 == 0:
            dt = (time.time() - tw) / 20; tw = time.time()
            print(f"step {step}/{N} loss {loss.item():.4f} kl {kl:.4f} dice {dice:.4f} "
                  f"teacher_fg {tfg*100:.2f}%  {dt*1000:.0f} ms/step  ETA {(N-step)*dt/3600:.2f} h", flush=True)
        if not args.smoke and step % 2000 == 0:
            torch.save(student.state_dict(), os.path.join(args.out, "student.pt"))
    if not args.smoke:
        torch.save(student.state_dict(), os.path.join(args.out, "student.pt"))
        with open(os.path.join(args.out, "DONE"), "w") as f:
            f.write(f"steps={N} elapsed_h={(time.time()-t0)/3600:.2f}\n")
        print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
