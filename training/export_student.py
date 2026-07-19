"""Export a trained student checkpoint to a browser-native single-file ONNX (fp16, fp32 I/O)."""
import os, argparse, json, torch
from student import Student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/nnlive_student/student.pt"))
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--out", default=os.path.expanduser("~/nnlive_student/student_128"))
    args = ap.parse_args()

    m = Student(cin=8, cout=2, base=args.base)
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    m.eval()
    dummy = torch.zeros(1, 8, args.patch, args.patch, args.patch)
    fp32 = args.out + "_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(m, dummy, fp32, input_names=["input"], output_names=["logits"],
                          opset_version=18, do_constant_folding=True)

    import onnx
    from onnxconverter_common import float16
    mo = onnx.load(fp32)
    ops = {}
    for n in mo.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print("OP_HISTOGRAM", json.dumps(dict(sorted(ops.items(), key=lambda x: -x[1]))))
    # sanity: none of the WebGPU-unsupported ops should be present
    bad = {"ConvTranspose", "AveragePool", "MaxPool"} & set(ops)
    print("BROWSER_NATIVE", "OK" if not bad else f"WARNING contains {bad}")
    fp16 = args.out + "_fp16.onnx"
    onnx.save_model(float16.convert_float_to_float16(mo, keep_io_types=True), fp16,
                    save_as_external_data=False)
    print("EXPORTED", fp16, round(os.path.getsize(fp16) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
