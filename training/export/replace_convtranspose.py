"""Replace 3D ConvTranspose(k2,s2) with a math-identical pixel-shuffle — WebGPU-safe.

ORT-Web WebGPU (1.27) has no ConvTranspose3d. A non-overlapping k2/s2 transposed conv is exactly:
  Y[co, 2d+a, 2h+b, 2w+c] = sum_ci X[ci,d,h,w] * W[ci,co,a,b,c] + bias[co]
i.e. a 1x1x1 Conv producing Co*8 channels (packed as co*8 + a*4+b*2+c), followed by a 2x2x2
pixel-shuffle interleave (Reshape -> 8D Transpose -> Reshape) — all WebGPU-supported ops
(verified: the 8D Transpose runs on the WebGPU EP). Static shapes are baked per stage from shape
inference, so no Shape/Gather ops are needed.

Input: net_192_slice_fp32.onnx (AvgPool already replaced). Output: net_192_webgpu_fp32.onnx (+fp16).
"""
import os, numpy as np, onnx
from onnx import helper, numpy_helper

TAG = os.environ.get("NNLIVE_TAG", "192")
SRC = os.path.expanduser(f"~/nnlive_export/net_{TAG}_slice_fp32.onnx")
DST = os.path.expanduser(f"~/nnlive_export/net_{TAG}_webgpu_fp32.onnx")

m = onnx.shape_inference.infer_shapes(onnx.load(SRC))
g = m.graph
shape_of = {}
for vi in list(g.value_info) + list(g.input) + list(g.output):
    d = vi.type.tensor_type.shape.dim
    if len(d) == 5 and all(x.HasField("dim_value") for x in d):
        shape_of[vi.name] = [x.dim_value for x in d]
inits = {i.name: i for i in g.initializer}

new_nodes, new_inits, cnt = [], [], 0
for node in g.node:
    if node.op_type != "ConvTranspose":
        new_nodes.append(node); continue
    X, Wn, Y = node.input[0], node.input[1], node.output[0]
    W = numpy_helper.to_array(inits[Wn])                       # [Ci, Co, 2,2,2]
    Ci, Co = W.shape[0], W.shape[1]
    bias = numpy_helper.to_array(inits[node.input[2]]) if len(node.input) > 2 else np.zeros(Co, W.dtype)
    D, H, Wd = shape_of[X][2:5]
    p = node.name
    # 1x1x1 Conv weight/bias packed as out-channel = co*8 + a*4+b*2+c
    nw = W.transpose(1, 2, 3, 4, 0).reshape(Co * 8, Ci, 1, 1, 1).astype(W.dtype)
    nb = np.repeat(bias, 8).astype(bias.dtype)
    new_inits += [numpy_helper.from_array(nw, p + "_w"), numpy_helper.from_array(nb, p + "_b")]
    new_inits += [
        numpy_helper.from_array(np.array([1, Co, 2, 2, 2, D, H, Wd], np.int64), p + "_s1"),
        numpy_helper.from_array(np.array([1, Co, 2 * D, 2 * H, 2 * Wd], np.int64), p + "_s2"),
    ]
    new_nodes += [
        helper.make_node("Conv", [X, p + "_w", p + "_b"], [p + "_c"], name=p + "_conv1x1", kernel_shape=[1, 1, 1]),
        helper.make_node("Reshape", [p + "_c", p + "_s1"], [p + "_r1"]),
        helper.make_node("Transpose", [p + "_r1"], [p + "_t1"], perm=[0, 1, 5, 2, 6, 3, 7, 4]),
        helper.make_node("Reshape", [p + "_t1", p + "_s2"], [Y]),
    ]
    cnt += 1

del g.node[:]; g.node.extend(new_nodes); g.initializer.extend(new_inits)
m = onnx.shape_inference.infer_shapes(m); onnx.checker.check_model(m)
onnx.save_model(m, DST, save_as_external_data=False)
print(f"REPLACED {cnt} ConvTranspose -> pixel-shuffle; saved {DST} ({os.path.getsize(DST)/1e6:.0f} MB)")

from onnxconverter_common import float16
DST16 = os.path.expanduser(f"~/nnlive_export/net_{TAG}_webgpu_fp16.onnx")
onnx.save_model(float16.convert_float_to_float16(onnx.load(DST), keep_io_types=True), DST16,
                save_as_external_data=False)
print(f"fp16 saved {DST16} ({os.path.getsize(DST16)/1e6:.0f} MB)")
