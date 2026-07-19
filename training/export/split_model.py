"""Split a static ONNX graph into N sequential sub-models for chunked WebGPU execution.

Each sub-model runs as a separate short GPU submission; the JS harness chains them keeping
intermediates on-GPU (IO-binding) and drains the GPU between chunks so WindowServer isn't starved.
Cuts handle U-Net skip connections by carrying the FULL set of tensors crossing each boundary.

Emits sub_000.onnx ... sub_{N-1}.onnx + chunks.json (the wiring: each chunk's input/output tensors).
"""
import os, json, argparse, onnx
from onnx import shape_inference
from onnx.utils import extract_model

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.expanduser("~/nnlive_export/net_192_webgpu_fp32.onnx"))
ap.add_argument("--out", default=os.path.expanduser("~/nnlive_chunks"))
ap.add_argument("--chunks", type=int, default=16)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

m = shape_inference.infer_shapes(onnx.load(args.src))
g = m.graph
nodes = list(g.node)
init_names = {i.name for i in g.initializer}
graph_inputs = [i.name for i in g.input if i.name not in init_names]
graph_outputs = [o.name for o in g.output]

# producer index per tensor, and every consumer index
producer = {}
consumers = {}
for idx, n in enumerate(nodes):
    for o in n.output:
        producer[o] = idx
    for inp in n.input:
        consumers.setdefault(inp, []).append(idx)

def live_across(b):
    """Tensors produced by nodes[0..b] and consumed by some node > b (the edge cut set)."""
    live = set()
    for t, pidx in producer.items():
        if pidx <= b and any(c > b for c in consumers.get(t, [])):
            live.add(t)
    return sorted(live)

# even boundaries in node-index space
N = args.chunks
bounds = [round((k + 1) * len(nodes) / N) - 1 for k in range(N - 1)]

cuts = [graph_inputs] + [live_across(b) for b in bounds] + [graph_outputs]
manifest = {"src": os.path.basename(args.src), "chunks": []}
for k in range(N):
    ins, outs = cuts[k], cuts[k + 1]
    path = os.path.join(args.out, f"sub_{k:03d}.onnx")
    extract_model(args.src, path, ins, outs)
    manifest["chunks"].append({"file": f"sub_{k:03d}.onnx", "inputs": ins, "outputs": outs,
                               "bytes": os.path.getsize(path)})
    print(f"chunk {k}: {len(ins)} in -> {len(outs)} out  ({os.path.getsize(path)/1e6:.1f} MB)")

json.dump(manifest, open(os.path.join(args.out, "chunks.json"), "w"), indent=2)
print(f"wrote {N} sub-models + chunks.json to {args.out}")
