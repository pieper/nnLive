"""M0 spike: export the nnInteractive network to ONNX and report its op histogram.

The network is a static-shape module: (1, C, *patch_size) float32 -> (1, n_classes, *patch_size).
Everything intricate (crop/resize/dilate/autozoom) is orchestration *around* this call and is
ported+validated separately against the Python session as oracle. This script only tests the
network's exportability and operator set — the gate for running it on ORT-Web WebGPU.
"""
import os, json, torch
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

WEIGHTS = os.path.expanduser("~/.nninteractive_weights/nnInteractive_v1.0")
OUT = os.path.expanduser("~/nnlive_export")
os.makedirs(OUT, exist_ok=True)

# Size-parametrized so the same pipeline exports 128^3 (memory-safe on 16GB) or 192^3 (native).
# Defaults preserve the original behavior exactly. NNLIVE_DEV=cpu lets us re-export without touching
# the training GPU. NNLIVE_SIZE=0 keeps the model's native patch_size.
DEV = os.environ.get("NNLIVE_DEV", "cuda:0")
SIZE = int(os.environ.get("NNLIVE_SIZE", "0"))
TAG = os.environ.get("NNLIVE_TAG", "192")

sess = nnInteractiveInferenceSession(
    device=torch.device(DEV),
    use_torch_compile=False,
    verbose=False,
    torch_n_threads=os.cpu_count(),
    do_autozoom=True,
    interactions_storage="auto",
)
sess.initialize_from_trained_model_folder(WEIGHTS)

net = sess.network
net = getattr(net, "_orig_mod", net)   # unwrap torch.compile if present
net.eval()
# nnU-Net returns a single tensor at inference; make sure deep supervision is off.
if hasattr(net, "decoder") and hasattr(net.decoder, "deep_supervision"):
    net.decoder.deep_supervision = False

first_conv = next(m for m in net.modules() if isinstance(m, torch.nn.Conv3d))
C = first_conv.in_channels
ps = [int(x) for x in sess.configuration_manager.patch_size]
if SIZE:
    ps = [SIZE, SIZE, SIZE]                      # override to a browser-affordable patch (fully convolutional)
print("INPUT_CHANNELS", C, "PATCH_SIZE", ps, "DEV", DEV, "TAG", TAG)

dummy = torch.zeros((1, C, *ps), dtype=torch.float32, device=DEV)
with torch.no_grad():
    out = net(dummy)
print("FORWARD_OK output_shape", list((out[0] if isinstance(out,(list,tuple)) else out).shape))

onnx_path = os.path.join(OUT, f"nninteractive_net_{TAG}.onnx")
with torch.no_grad():
    torch.onnx.export(
        net, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        opset_version=18, do_constant_folding=True,
    )
sz = os.path.getsize(onnx_path) / 1e6
print("EXPORTED", onnx_path, round(sz, 1), "MB")

import onnx
m = onnx.load(onnx_path)
ops = {}
for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("OP_HISTOGRAM", json.dumps(dict(sorted(ops.items(), key=lambda x: -x[1]))))
print("N_NODES", len(m.graph.node))
