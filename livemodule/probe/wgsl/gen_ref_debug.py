"""Emit ORT reference for a spanning set of intermediate tensors (first node of each op type +
a few spread through the graph) so the JS runtime can find the FIRST divergent op."""
import os, json, numpy as np, onnx, onnxruntime as ort
SRC="/private/tmp/claude-501/-Users-pieper-slicer-SlicerLive/e7c441a0-d53d-4b62-8d64-85e357888d8c/scratchpad/pathA_onnx/perclick_64.onnx"
DST="/Users/pieper/slicer/nnLive/livemodule/probe/models/pathA/ref/dbg"; os.makedirs(DST,exist_ok=True)
g=json.load(open("/private/tmp/claude-501/-Users-pieper-slicer-SlicerLive/e7c441a0-d53d-4b62-8d64-85e357888d8c/scratchpad/pathA_onnx/perclick_64.graph.json"))
# pick first-of-each-op + spread by node index
picks={};
for i,nd in enumerate(g["nodes"]):
    if nd["op"] not in picks: picks[nd["op"]]=nd["out"][0]
for i in range(28,90,3): picks[f"node{i:03d}"]=g["nodes"][i]["out"][0]
for i in (150,220): picks[f"node{i}"]=g["nodes"][i]["out"][0]
names=list(dict.fromkeys(picks.values()))
m=onnx.load(SRC)
have={o.name for o in m.graph.output}
for nm in names:
    if nm not in have: m.graph.output.extend([onnx.helper.make_empty_tensor_value_info(nm)])
tmp=SRC+".dbg.onnx"; onnx.save(m,tmp)
s=ort.InferenceSession(tmp,providers=["CPUExecutionProvider"])
np.random.seed(7); feeds={i.name:(np.random.randn(*i.shape).astype(np.float32)*0.5) for i in s.get_inputs()}
outs=s.run(names,feeds)
san=lambda x:x.replace('/','_').replace('.','_')
manifest={}
for nm,arr in zip(names,outs):
    fn=san(nm)+".f32"; arr.astype(np.float32).tofile(os.path.join(DST,fn)); manifest[nm]={"file":fn,"shape":list(arr.shape)}
json.dump({"picks":picks,"manifest":manifest},open(os.path.join(DST,"dbg.json"),"w"))
print("dumped", len(names), "intermediates:")
for op,nm in picks.items(): print(f"  {op:22s} -> {nm}")
