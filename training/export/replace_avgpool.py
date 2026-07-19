"""Graph surgery: replace 3D AveragePool nodes with equivalent depthwise strided Conv.

ORT-Web's WebGPU EP does not implement 3D pooling (NHWC, kernelShape.length>2). An average
pool over C channels with kernel k and stride s is exactly a depthwise (groups=C) Conv3D whose
weights are all 1/prod(k). Convolution IS covered by the WebGPU EP, so this keeps the whole
network on the GPU with no CPU fallback and no change to the math.
"""
import os, numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto

SRC = os.path.expanduser("~/nnlive_export/nninteractive_net_192.onnx")  # fp32 + external data
DST = os.path.expanduser("~/nnlive_export/net_192_conv_fp32.onnx")

m = onnx.load(SRC)
m = onnx.shape_inference.infer_shapes(m)
g = m.graph

# channel count per tensor from shape inference (dim 1 = channels, NCHW-ish 5D: N C D H W)
chan = {}
for vi in list(g.value_info) + list(g.input) + list(g.output):
    d = vi.type.tensor_type.shape.dim
    if len(d) == 5 and d[1].HasField("dim_value"):
        chan[vi.name] = d[1].dim_value

new_nodes, n_replaced, init = [], 0, []
for node in g.node:
    if node.op_type != "AveragePool":
        new_nodes.append(node); continue
    attrs = {a.name: a for a in node.attribute}
    k = list(attrs["kernel_shape"].ints)
    s = list(attrs["strides"].ints) if "strides" in attrs else [1]*len(k)
    pads = list(attrs["pads"].ints) if "pads" in attrs else [0]*(2*len(k))
    C = chan.get(node.input[0])
    if C is None or len(k) != 3:
        new_nodes.append(node); continue
    w = np.full((C, 1, *k), 1.0/np.prod(k), dtype=np.float32)
    wname = node.output[0] + "_avgw"
    init.append(numpy_helper.from_array(w, wname))
    conv = helper.make_node(
        "Conv", inputs=[node.input[0], wname], outputs=list(node.output),
        name=node.name + "_as_conv", kernel_shape=k, strides=s, pads=pads, group=C,
    )
    new_nodes.append(conv); n_replaced += 1

del g.node[:]; g.node.extend(new_nodes); g.initializer.extend(init)
m = onnx.shape_inference.infer_shapes(m)
onnx.checker.check_model(m)
onnx.save_model(m, DST, save_as_external_data=False)
print(f"REPLACED {n_replaced} AveragePool -> depthwise Conv; saved {DST} "
      f"({os.path.getsize(DST)/1e6:.0f} MB)")

# fp16 (fp32 I/O) version for the browser
from onnxconverter_common import float16
m16 = float16.convert_float_to_float16(onnx.load(DST), keep_io_types=True)
DST16 = os.path.expanduser("~/nnlive_export/net_192_conv_fp16.onnx")
onnx.save_model(m16, DST16, save_as_external_data=False)
print(f"fp16 saved {DST16} ({os.path.getsize(DST16)/1e6:.0f} MB)")
