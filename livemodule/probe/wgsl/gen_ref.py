"""Generate a fixed-input ORT reference for the perclick graph, for validating the custom WGSL runtime.
Saves s0/s1/inter (f32) + logits_ref (f32) as raw .bin into the probe dir."""
import os, sys, json, numpy as np, onnxruntime as ort
SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "/private/tmp/claude-501/-Users-pieper-slicer-SlicerLive/e7c441a0-d53d-4b62-8d64-85e357888d8c/scratchpad/pathA_onnx/perclick_64.onnx"
DST = "/Users/pieper/slicer/nnLive/livemodule/probe/models/pathA/ref"
os.makedirs(DST, exist_ok=True)
s = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"])
np.random.seed(7)
feeds = {}
for i in s.get_inputs():
    a = np.random.randn(*i.shape).astype(np.float32) * 0.5
    feeds[i.name] = a
    a.tofile(os.path.join(DST, f"{i.name}.f32"))
out = s.run(None, feeds)[0].astype(np.float32)
out.tofile(os.path.join(DST, "logits_ref.f32"))
meta = {"inputs": {i.name: list(i.shape) for i in s.get_inputs()}, "out": list(out.shape)}
json.dump(meta, open(os.path.join(DST, "ref_meta.json"), "w"))
print("ref written:", meta, "| logits mean/std", float(out.mean()), float(out.std()))
