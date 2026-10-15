"""Export a real-world RPS student to ONNX and fully-integer INT8 TFLite (same steps and checks as
the research's qakd.deploy.export, but reading realworld/runs/ and writing realworld/exports/).

    python realworld/export.py --run student_mnv3s_kd_s0

checkpoint -> ONNX opset 17 (static batch 1) -> ONNX Runtime parity -> onnx2tf, keeping three
TFLite variants, each scored on the real test split and the rendered RPS test set:
  model_fp32.tflite  float reference
  model_drq.tflite   dynamic-range quantization: INT8 weights, float activations — the DEPLOYED
                     model. The ImageNet-pretrained MobileNetV3-Small's activation ranges do not
                     survive 8-bit post-training quantization (hard-swish / squeeze-excitation;
                     torchvision ships its quantized MobileNetV3 via quantization-aware training
                     for this reason): full INT8 fell 92% -> 36% on the real test while DRQ
                     held 90%, at about the same file size.
  model_int8.tflite  full-integer INT8, per-channel, calibrated on 256 un-augmented training
                     images — kept for the record, not deployed.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_rps  # noqa: E402
from train import RUNS, Student, evaluate  # noqa: E402
from qakd.deploy import purity_check  # noqa: E402

EXPORTS = os.path.join(data_rps.REPO, "realworld", "exports")
N_CALIBRATION = 256


def _arrays(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.permute(0, 2, 3, 1).numpy().astype(np.float32))
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def _calibration():
    split = data_rps.build_split()
    real = data_rps.RealFrames(split["train"], data_rps.eval_transform())
    rendered = data_rps.RenderedRPS(data_rps.rps_entries("train"), data_rps.eval_transform())
    n_r = min(len(real), N_CALIBRATION // 2)
    idx_r = np.linspace(0, len(rendered) - 1, N_CALIBRATION - n_r).round().astype(int)
    xs = [real[i][0] for i in range(n_r)] + [rendered[int(i)][0] for i in idx_r]
    return torch.stack(xs).permute(0, 2, 3, 1).numpy().astype(np.float32)


def _tflite_eval(path, x):
    """Logits from any variant: int8 I/O is (de)quantized exactly as the research export does."""
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    s, z = inp["quantization"]
    os_, oz = out["quantization"]
    logits = []
    for xi in x:
        xin = (np.clip(np.round(xi / s + z), -128, 127).astype(np.int8) if inp["dtype"] == np.int8
               else xi.astype(np.float32))[None]
        interp.set_tensor(inp["index"], xin)
        interp.invoke()
        o = interp.get_tensor(out["index"]).astype(np.float32)
        logits.append((o - oz) * os_ if out["dtype"] == np.int8 else o)
    return np.concatenate(logits), interp


def _metrics(y, pred):
    from sklearn.metrics import f1_score
    present = sorted(set(y.tolist()))
    return {"accuracy": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro", labels=present))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    out_dir = os.path.join(EXPORTS, args.run)
    os.makedirs(out_dir, exist_ok=True)

    model = Student(len(data_rps.CLASSES), pretrained=False)
    state = torch.load(os.path.join(RUNS, args.run, "best.pt"), map_location="cpu")
    purity_check.check_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    loaders, _ = data_rps.build_loaders(64, num_workers=0)
    tests = {k: _arrays(loaders[k]) for k in ("test", "rps_test")}
    fp32 = {k: evaluate(model, loaders[k]) for k in ("test", "rps_test")}

    onnx_path = os.path.join(out_dir, "model.onnx")
    torch.onnx.export(model, (torch.zeros(1, 3, data_rps.IMAGE_SIZE, data_rps.IMAGE_SIZE),), onnx_path,
                      opset_version=17, dynamo=False, input_names=["input"], output_names=["logits"])
    import onnx
    import onnxruntime as ort
    onnx_ops = purity_check.check_onnx(onnx.load(onnx_path))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x_test = tests["test"][0]
    ort_logits = np.concatenate([sess.run(None, {"input": xi.transpose(2, 0, 1)[None]})[0] for xi in x_test])
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(x_test).permute(0, 3, 1, 2)).numpy()
    parity = {"max_abs_logit_diff": float(np.abs(ort_logits - torch_logits).max()),
              "argmax_agreement": float((ort_logits.argmax(1) == torch_logits.argmax(1)).mean())}
    if parity["argmax_agreement"] < 1.0 or parity["max_abs_logit_diff"] > 1e-3:
        raise purity_check.PurityError(f"ONNX parity failed: {parity}")

    calib_path = os.path.join(out_dir, "calibration_nhwc.npy")
    np.save(calib_path, _calibration())
    tf_dir = os.path.join(out_dir, "tf")
    import onnx2tf
    onnx2tf.convert(input_onnx_file_path=onnx_path, output_folder_path=tf_dir,
                    output_integer_quantized_tflite=True, output_dynamic_range_quantized_tflite=True,
                    quant_type="per-channel", tflite_backend="tf_converter",
                    custom_input_op_name_np_data_path=[["input", calib_path, [[[[0.0, 0.0, 0.0]]]], [[[[1.0, 1.0, 1.0]]]]]],
                    input_quant_dtype="int8", output_quant_dtype="int8", non_verbose=True)
    variants = {}
    for key, suffix in (("fp32", "_float32"), ("drq", "_dynamic_range_quant"), ("int8", "_full_integer_quant")):
        produced = glob.glob(os.path.join(tf_dir, f"*{suffix}.tflite"))
        if len(produced) != 1:
            raise purity_check.PurityError(f"expected one {suffix} TFLite file, found {produced}")
        variants[key] = os.path.join(out_dir, f"model_{key}.tflite")
        shutil.copyfile(produced[0], variants[key])
    shutil.rmtree(tf_dir, ignore_errors=True)
    os.remove(calib_path)
    scores, tflite_summary = {}, {}
    for key, path in variants.items():
        scores[key] = {}
        for k, (x, y) in tests.items():
            logits, interp = _tflite_eval(path, x)
            scores[key][k] = _metrics(y, logits.argmax(1))
        ops = sorted({op["op_name"] for op in interp._get_ops_details()})
        if any(o in ("CUSTOM", "FLEX") or o.startswith("FLEX") for o in ops):
            raise purity_check.PurityError(f"{key}: custom/flex ops {ops}")
        tflite_summary[key] = {"ops": ops, "size_kb": round(os.path.getsize(path) / 1024.0, 1)}
    tflite_summary["int8"].update(purity_check.check_tflite(_tflite_eval(variants["int8"], tests["test"][0][:1])[1]))

    report = {"run": args.run, "classes": data_rps.CLASSES, "image_size": data_rps.IMAGE_SIZE,
              "preprocessing": "whole frame resized to 128x128, ImageNet mean/std",
              "deployed": "model_drq.tflite",
              "torch_fp32": {k: {m: v[m] for m in ("accuracy", "macro_f1")} for k, v in fp32.items()},
              "tflite": scores, "tflite_files": tflite_summary,
              "onnx": {"opset": 17, "ops": onnx_ops, **parity}}
    with open(os.path.join(out_dir, "deploy.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("run", "torch_fp32", "tflite")}, indent=1))


if __name__ == "__main__":
    main()
