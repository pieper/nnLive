"""Replace 3D AveragePool(2x2x2, stride2) with strided-slice mean — WebGPU-safe.

ORT-Web WebGPU (1.27) supports Conv3D/ConvTranspose3D/InstanceNorm but NOT 3D AveragePool, and
its Conv3D has no grouped-conv support (so the depthwise-conv trick fails too). A non-overlapping
2x2x2 average pool is exactly the mean of the 8 corner-offset strided sub-samples:
    Y = (1/8) * sum_{a,b,c in {0,1}} X[:, :, a::2, b::2, c::2]
which uses only Slice + Add + Mul — all covered, no grouping, no high-rank reshape.
"""
import os, itertools, numpy as np, onnx
from onnx import helper, numpy_helper

TAG = os.environ.get("NNLIVE_TAG", "192")
SRC = os.path.expanduser(f"~/nnlive_export/nninteractive_net_{TAG}.onnx")
DST = os.path.expanduser(f"~/nnlive_export/net_{TAG}_slice_fp32.onnx")

m = onnx.load(SRC)
g = m.graph
INT_MAX = np.iinfo(np.int64).max

# Shared Slice params (axes/ends/steps identical everywhere; 8 starts combos reused across pools).
shared = {}
def init_once(name, arr):
    if name not in shared:
        g.initializer.append(numpy_helper.from_array(arr.astype(np.int64), name))
        shared[name] = True
    return name
init_once("sl_axes", np.array([2, 3, 4]))
init_once("sl_ends", np.array([INT_MAX, INT_MAX, INT_MAX]))
init_once("sl_steps", np.array([2, 2, 2]))
combos = list(itertools.product([0, 1], repeat=3))
for i, (a, b, c) in enumerate(combos):
    init_once(f"sl_start_{i}", np.array([a, b, c]))
g.initializer.append(numpy_helper.from_array(np.array(1.0/8.0, dtype=np.float32), "sl_eighth"))

new_nodes, n = [], 0
for node in g.node:
    if node.op_type != "AveragePool":
        new_nodes.append(node); continue
    X, Y = node.input[0], node.output[0]
    slice_outs = []
    for i in range(8):
        so = f"{Y}_sl{i}"
        new_nodes.append(helper.make_node(
            "Slice", [X, f"sl_start_{i}", "sl_ends", "sl_axes", "sl_steps"], [so],
            name=f"{node.name}_slice{i}"))
        slice_outs.append(so)
    cur = slice_outs[0]
    for i in range(1, 8):
        nxt = f"{Y}_sum{i}" if i < 7 else f"{Y}_sum"
        new_nodes.append(helper.make_node("Add", [cur, slice_outs[i]], [nxt],
                                          name=f"{node.name}_add{i}"))
        cur = nxt
    new_nodes.append(helper.make_node("Mul", [cur, "sl_eighth"], [Y], name=f"{node.name}_scale"))
    n += 1

del g.node[:]; g.node.extend(new_nodes)
m = onnx.shape_inference.infer_shapes(m)
onnx.checker.check_model(m)
onnx.save_model(m, DST, save_as_external_data=False)
print(f"REPLACED {n} AveragePool -> slice-mean; saved {DST} ({os.path.getsize(DST)/1e6:.0f} MB)")

from onnxconverter_common import float16
DST16 = os.path.expanduser(f"~/nnlive_export/net_{TAG}_slice_fp16.onnx")
onnx.save_model(float16.convert_float_to_float16(onnx.load(DST), keep_io_types=True),
                DST16, save_as_external_data=False)
print(f"fp16 saved {DST16} ({os.path.getsize(DST16)/1e6:.0f} MB)")
