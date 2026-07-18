"""Dump perclick ONNX -> graph.json + weights.bin (f16) for a minimal JS/WGSL executor.
Resolves shapes statically; drops shape-arithmetic (Shape/Slice/Constant/Cast/Gather/Unsqueeze
/Concat-of-1D) and bakes Resize target sizes. Keeps: Conv, InstanceNormalization, LeakyRelu,
Add, Concat(channel), AveragePool, MaxPool, Resize(nearest)."""
import os, sys, json, numpy as np, onnx
from onnx import shape_inference, numpy_helper

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "/private/tmp/claude-501/-Users-pieper-slicer-SlicerLive/e7c441a0-d53d-4b62-8d64-85e357888d8c/scratchpad/pathA_onnx/perclick_128.onnx"
OUT = os.path.dirname(SRC)
m = shape_inference.infer_shapes(onnx.load(SRC), data_prop=True)
g = m.graph
shp = {}
for vi in list(g.value_info) + list(g.input) + list(g.output):
    shp[vi.name] = [x.dim_value for x in vi.type.tensor_type.shape.dim]

# collect weight tensors: initializers + Constant node outputs
weights = {}
for i in g.initializer:
    weights[i.name] = numpy_helper.to_array(i)
for n in g.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                weights[n.output[0]] = numpy_helper.to_array(a.t)

KEEP = {"Conv", "InstanceNormalization", "LeakyRelu", "Add", "Concat", "AveragePool", "MaxPool", "Resize"}
nodes = []
for n in g.node:
    if n.op_type not in KEEP:
        continue
    if n.op_type == "Concat":
        # keep only channel concat of >=4D tensors (drop 1D shape concats)
        if any(len(shp.get(i, [])) < 4 for i in n.input):
            continue
    A = {a.name: a for a in n.attribute}
    def ai(name, d=0): return A[name].i if name in A else d
    def aints(name): return list(A[name].ints) if name in A else []
    def af(name, d=0.0): return A[name].f if name in A else d
    rec = {"op": n.op_type, "in": [x for x in n.input if x != ""], "out": list(n.output)}
    if n.op_type == "Conv":
        w = weights[n.input[1]]
        rec["Co"], rec["Ci"], rec["K"] = int(w.shape[0]), int(w.shape[1]), int(w.shape[2])
        rec["S"] = aints("strides")[0] if aints("strides") else 1
        rec["pad"] = aints("pads")[0] if aints("pads") else 0
        rec["bias"] = len(n.input) > 2 and n.input[2] != ""
    elif n.op_type == "InstanceNormalization":
        rec["eps"] = af("epsilon", 1e-5)
    elif n.op_type == "LeakyRelu":
        rec["alpha"] = af("alpha", 0.01)
    elif n.op_type == "Concat":
        rec["axis"] = ai("axis", 1)
    elif n.op_type in ("AveragePool", "MaxPool"):
        rec["kernel"] = aints("kernel_shape")[0]; rec["S"] = aints("strides")[0]
    elif n.op_type == "Resize":
        rec["mode"] = "nearest"; rec["in"] = [n.input[0]]            # keep only x; target read from output shape
    nodes.append(rec)

# emit weights.bin (f16) + offsets, only for weights actually consumed by kept nodes
used = set(i for nd in nodes for i in nd["in"])
blob = []; woff = {}; off = 0
for name in weights:
    if name not in used:
        continue
    arr = weights[name].astype(np.float16).ravel()
    woff[name] = {"offset": off, "numel": int(arr.size), "shape": [int(s) for s in weights[name].shape]}
    blob.append(arr); off += arr.size
base = os.path.splitext(os.path.basename(SRC))[0]
blobf16 = np.concatenate(blob) if blob else np.zeros(0, np.float16)
blobf16.tofile(os.path.join(OUT, f"{base}.weights.bin"))

# activation tensor shapes (everything with a shape that isn't a weight)
tensors = {k: v for k, v in shp.items() if k not in woff}
spec = {
    "inputs": [{"name": i.name, "shape": shp[i.name]} for i in g.input],
    "outputs": [{"name": o.name, "shape": shp[o.name]} for o in g.output],
    "weights": woff, "weightBytes": int(blobf16.nbytes),
    "tensors": tensors, "nodes": nodes,
}
json.dump(spec, open(os.path.join(OUT, f"{base}.graph.json"), "w"))
print(f"nodes kept: {len(nodes)} | weights: {len(woff)} ({blobf16.nbytes/1e6:.1f} MB) | tensors: {len(tensors)}")
import collections
print("kept ops:", dict(collections.Counter(n['op'] for n in nodes)))
print("conv variants (K,S,bias):", sorted(set((n['K'],n['S'],n['bias']) for n in nodes if n['op']=='Conv')))
print("wrote perclick.graph.json + perclick.weights.bin to", OUT)
