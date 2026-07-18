# Make perclick_192 onnx by overriding perclick_128's input dims to 192^3 and re-inferring
# shapes (no forward/trace). Then dump_graph.py turns it into graph.json + weights.
import onnx
from onnx import shape_inference
m = onnx.load("pathA_onnx/perclick_128.onnx")
g = m.graph
newdims = {"s0": [1, 32, 192, 192, 192], "s1": [1, 64, 96, 96, 96], "inter": [1, 7, 192, 192, 192]}
for i in g.input:
    if i.name in newdims:
        d = i.type.tensor_type.shape.dim
        for k, v in enumerate(newdims[i.name]):
            d[k].dim_value = v
del g.value_info[:]                                   # drop stale 128^3 intermediate shapes
for o in g.output:
    o.type.tensor_type.shape.Clear()                  # let inference recompute output shape
m2 = shape_inference.infer_shapes(m, data_prop=True)
onnx.save(m2, "pathA_onnx/perclick_192.onnx")
for o in m2.graph.output:
    print("output", o.name, [x.dim_value for x in o.type.tensor_type.shape.dim])
