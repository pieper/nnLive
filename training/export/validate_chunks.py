"""Confirm the chained sub-models are numerically identical to the whole model (ORT-CPU)."""
import os, json, time, numpy as np, onnxruntime as ort

D = os.path.expanduser("~/nnlive_chunks")
man = json.load(open(os.path.join(D, "chunks.json")))
WHOLE = os.path.expanduser("~/nnlive_export/net_192_webgpu_fp32.onnx")

whole = ort.InferenceSession(WHOLE, providers=["CPUExecutionProvider"])
iname = whole.get_inputs()[0].name
x = np.random.randn(1, 8, 192, 192, 192).astype(np.float32)

t = time.time(); y_whole = whole.run(None, {iname: x})[0]
print(f"whole forward {time.time()-t:.1f}s", flush=True)

tensors = {iname: x}
t = time.time()
for c in man["chunks"]:
    s = ort.InferenceSession(os.path.join(D, c["file"]), providers=["CPUExecutionProvider"])
    outs = s.run(c["outputs"], {n: tensors[n] for n in c["inputs"]})
    tensors.update(zip(c["outputs"], outs))
print(f"chained forward {time.time()-t:.1f}s", flush=True)

y_chain = tensors[man["chunks"][-1]["outputs"][0]]
d = float(np.abs(y_whole - y_chain).max())
print(f"CHUNK_VALIDATION max_abs_diff {d}", flush=True)
open(os.path.join(D, "validation.txt"), "w").write(f"max_abs_diff={d}\n")
