"""Export the trained Path A model to ONNX, split for encode-once/decode-per-click:
  trunk.onnx    : image[1,1,P,P,P] -> s0,s1        (frozen image trunk, run ONCE per volume)
  perclick.onnx : s0,s1,inter[1,7] -> logits[1,2]  (injectors + deep stages + slim decoder, per click)
Validates ONNX (CPU) == torch on a random input. Self-contained (no distill_real import)."""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnInteractive, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "pathA_ckpt")
WEIGHTS = "/Users/pieper/slicer/latest/nnInteractive/.nninteractive_weights/nnInteractive_v1.0"
OUT = os.path.join(HERE, "pathA_onnx"); os.makedirs(OUT, exist_ok=True)
FEATS = [32, 64, 128, 256, 320, 320]
P = int(sys.argv[1]) if len(sys.argv) > 1 else 128
TAG = f"_{P}"

# ---- rebuild teacher net (for its trained trunk+deep weights) ----
plans = PlansManager(load_json(join(WEIGHTS, "plans.json"))); ds = load_json(join(WEIGHTS, "dataset.json"))
cm = plans.get_configuration("3d_fullres_ps192")
tr = recursive_find_python_class(join(nnInteractive.__path__[0], "trainer"),
                                 "nnInteractiveTrainer_stub", "nnInteractive.trainer")
teacher = tr.build_network_architecture(cm.network_arch_class_name, cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import, 1, 2, enable_deep_supervision=False).eval()

def mk_inject(c, hidden=96):
    m = nn.Sequential(nn.Conv3d(c + 7, hidden, 3, padding=1), nn.InstanceNorm3d(hidden), nn.LeakyReLU(0.01, True),
                      nn.Conv3d(hidden, hidden, 3, padding=1), nn.InstanceNorm3d(hidden), nn.LeakyReLU(0.01, True),
                      nn.Conv3d(hidden, c, 3, padding=1))
    return m

class PathAModel(nn.Module):   # matches distill_pathA (multiscale, hidden 96)
    def __init__(self, teacher):
        super().__init__()
        import copy; net = copy.deepcopy(teacher)
        self.enc, self.dec = net.encoder, net.decoder
        self.inject = mk_inject(FEATS[1]); self.inject2 = mk_inject(FEATS[2]); self.inject3 = mk_inject(FEATS[3])

class SlimDecoder(nn.Module):  # matches distill_slimdec (width 0.5)
    def __init__(self, skip_ch=FEATS, width=0.5, nclass=2, cmin=16):
        super().__init__()
        self.cw = [max(cmin, round(width * c)) for c in skip_ch]
        self.start = nn.Conv3d(skip_ch[5], self.cw[5], 1)
        self.up, self.lat, self.fuse = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for lvl in range(4, -1, -1):
            self.up.append(nn.Conv3d(self.cw[lvl + 1], self.cw[lvl], 1))
            self.lat.append(nn.Conv3d(skip_ch[lvl], self.cw[lvl], 1))
            self.fuse.append(nn.Sequential(nn.Conv3d(self.cw[lvl], self.cw[lvl], 3, padding=1),
                                           nn.InstanceNorm3d(self.cw[lvl]), nn.LeakyReLU(0.01, True)))
        self.head = nn.Conv3d(self.cw[0], nclass, 1)
    def forward(self, skips):
        x = self.start(skips[5])
        for i, lvl in enumerate(range(4, -1, -1)):
            x = F.interpolate(x, size=skips[lvl].shape[2:], mode="nearest")
            x = self.up[i](x) + self.lat[i](skips[lvl]); x = self.fuse[i](x)
        return self.head(x)

pathA = PathAModel(teacher).eval()
sd = torch.load(os.path.join(CKPT, "r2_pathA_model.pt"), map_location="cpu")
msd = pathA.state_dict()
pathA.load_state_dict({k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}, strict=False)
slim = SlimDecoder(width=0.5).eval()
slim.load_state_dict(torch.load(os.path.join(CKPT, "slimdec_w0.5.pt"), map_location="cpu"))
print("loaded r2 + slim decoder")

class Trunk(nn.Module):
    def __init__(s, p): super().__init__(); s.enc = p.enc
    def forward(s, image):
        z = image.new_zeros(image.shape[0], 7, *image.shape[2:])
        img8 = torch.cat([image, z], 1)
        h = s.enc.stem(img8); s0 = s.enc.stages[0](h); s1 = s.enc.stages[1](s0)
        return s0, s1

class PerClick(nn.Module):
    def __init__(s, p, slim):
        super().__init__(); s.enc = p.enc; s.inj = p.inject; s.inj2 = p.inject2; s.inj3 = p.inject3; s.slim = slim
    def forward(s, s0, s1, inter):
        ren = lambda t: F.adaptive_max_pool3d(inter, t.shape[2:])
        s1p = s1 + s.inj(torch.cat([s1, ren(s1)], 1))
        s2 = s.enc.stages[2](s1p); s2 = s2 + s.inj2(torch.cat([s2, ren(s2)], 1))
        s3 = s.enc.stages[3](s2); s3 = s3 + s.inj3(torch.cat([s3, ren(s3)], 1))
        s4 = s.enc.stages[4](s3); s5 = s.enc.stages[5](s4)
        return s.slim([s0, s1, s2, s3, s4, s5])

trunk = Trunk(pathA).eval(); perclick = PerClick(pathA, slim).eval()

# fast mode: export ONLY perclick at P, tracing with ZERO inputs (skip slow trunk forward / validation / fp16)
if len(sys.argv) > 2 and sys.argv[2] == "fast":
    s0 = torch.zeros(1, 32, P, P, P); s1 = torch.zeros(1, 64, P // 2, P // 2, P // 2); inter = torch.zeros(1, 7, P, P, P)
    torch.onnx.export(perclick, (s0, s1, inter), join(OUT, f"perclick{TAG}.onnx"),
        input_names=["s0", "s1", "inter"], output_names=["logits"], opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"FAST: wrote perclick{TAG}.onnx"); sys.exit(0)

# trunk8 mode: export trunk taking an 8-channel input directly (image in ch0, zeros ch1-7) -> s0,s1. Zero-trace.
if len(sys.argv) > 2 and sys.argv[2] == "trunk8":
    class Trunk8(nn.Module):
        def __init__(s, p): super().__init__(); s.enc = p.enc
        def forward(s, img8):
            h = s.enc.stem(img8); a = s.enc.stages[0](h); b = s.enc.stages[1](a); return a, b
    img8 = torch.zeros(1, 8, P, P, P)
    torch.onnx.export(Trunk8(pathA).eval(), (img8,), join(OUT, f"trunk8{TAG}.onnx"),
        input_names=["img8"], output_names=["s0", "s1"], opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"TRUNK8: wrote trunk8{TAG}.onnx"); sys.exit(0)

# reference forward
torch.manual_seed(0)
img = torch.randn(1, 1, P, P, P); inter = (torch.rand(1, 7, P, P, P) > 0.98).float()
with torch.no_grad():
    s0, s1 = trunk(img); ref_logits = perclick(s0, s1, inter)
print(f"shapes: s0{list(s0.shape)} s1{list(s1.shape)} logits{list(ref_logits.shape)}")

# ALSO export a COMBINED single-graph model (image+inter -> logits) for single-context graph capture
class Full(nn.Module):
    def __init__(s, tr, pc): super().__init__(); s.tr = tr; s.pc = pc
    def forward(s, image, inter):
        s0, s1 = s.tr(image); return s.pc(s0, s1, inter)
full = Full(trunk, perclick).eval()

TRUNK, PERCLICK, FULL = f"trunk{TAG}.onnx", f"perclick{TAG}.onnx", f"full{TAG}.onnx"
# export
torch.onnx.export(trunk, (img,), join(OUT, TRUNK), input_names=["image"],
    output_names=["s0", "s1"], opset_version=17, do_constant_folding=True, dynamo=False)
torch.onnx.export(perclick, (s0, s1, inter), join(OUT, PERCLICK),
    input_names=["s0", "s1", "inter"], output_names=["logits"], opset_version=17,
    do_constant_folding=True, dynamo=False)
torch.onnx.export(full, (img, inter), join(OUT, FULL), input_names=["image", "inter"],
    output_names=["logits"], opset_version=17, do_constant_folding=True, dynamo=False)
for n in [TRUNK, PERCLICK, FULL]:
    print(f"  {n}: {os.path.getsize(join(OUT, n))/1e6:.1f} MB")

# validate ONNX (CPU) == torch
st = ort.InferenceSession(join(OUT, TRUNK), providers=["CPUExecutionProvider"])
sp = ort.InferenceSession(join(OUT, PERCLICK), providers=["CPUExecutionProvider"])
o_s0, o_s1 = st.run(None, {"image": img.numpy()})
o_log = sp.run(None, {"s0": o_s0, "s1": o_s1, "inter": inter.numpy()})[0]
d_log = np.abs(o_log - ref_logits.numpy()).max()
sign_agree = (np.sign(o_log[:, 1] - o_log[:, 0]) == np.sign(ref_logits.numpy()[:, 1] - ref_logits.numpy()[:, 0])).mean()
print(f"VALIDATE perclick max|diff| {d_log:.2e} | mask sign-agree {sign_agree*100:.2f}%")

# fp16 (keep_io_types=False avoids the Cast-mismatch bug) + copy to probe dir
from onnxconverter_common import float16
import onnx, shutil
PROBE = "/Users/pieper/slicer/nnLive/livemodule/probe/models/pathA"
os.makedirs(PROBE, exist_ok=True)
for n in [TRUNK, PERCLICK, FULL]:
    base = n[:-5]
    m16 = float16.convert_float_to_float16(onnx.load(join(OUT, n)), keep_io_types=False)
    onnx.save(m16, join(OUT, base + "_fp16.onnx"))
    shutil.copy(join(OUT, base + "_fp16.onnx"), join(PROBE, base + "_fp16.onnx"))
    print(f"  {base}_fp16.onnx: {os.path.getsize(join(OUT, base + '_fp16.onnx'))/1e6:.1f} MB -> probe")
