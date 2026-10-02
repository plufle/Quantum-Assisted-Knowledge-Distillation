"""Export a trained student to ONNX and fully-integer INT8 TFLite, verifying at every step.

    python -m qakd.deploy.export --run trashnet__mobilenetv2_035__kd__s0

Pipeline (CLAUDE.md Deploy):
  checkpoint -> purity_check -> fp32 reload parity (must reproduce results/{run}/metrics.json)
  -> ONNX opset 17 -> purity_check -> ONNX Runtime parity -> TFLite INT8 per-channel
  (calibrated on training images) -> purity_check (fully integer, per-channel, no custom ops)
  -> INT8 accuracy on the same test split with the same preprocessing.

Outputs go to exports/{run_id}/ — results/{run_id}/ is immutable and is only ever read.
Projection heads and the teacher are never part of the student checkpoint, so "strip
projection + quantum modules" is satisfied by construction and *verified* by the purity check.
"""
import argparse
import glob
import json
import os
import shutil
import time

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from sklearn.metrics import accuracy_score, f1_score

import qakd.models.students  # noqa: F401 — registers STUDENTS
from qakd.data.datasets import build_bn_calibration_loader, build_rps25_loaders, build_trashnet_loaders
from qakd.deploy import purity_check
from qakd.logger import logger
from qakd.models.registry import build_student_from_cfg

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
EXPORT_ROOT = os.path.join(REPO_ROOT, "exports")
ONNX_OPSET = 17
N_CALIBRATION = 256
# fp32 reload must reproduce the recorded test score — anything beyond float noise means the
# export path is not evaluating the model the trainer evaluated.
FP32_TOLERANCE = 1e-6

_LOADERS = {"trashnet": build_trashnet_loaders, "rps_25": build_rps25_loaders}


def _parse_run_id(run_id):
    dataset, student, method, seed = run_id.split("__")
    return dataset, student, method, int(seed.lstrip("s"))


def _compose(dataset, student):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="config", overrides=[f"dataset={dataset}", f"student={student}"])


def _metrics(labels, preds):
    return {"macro_f1": float(f1_score(labels, preds, average="macro")),
            "top1_accuracy": float(accuracy_score(labels, preds))}


def _torch_eval(model, loader):
    logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x).numpy())
            labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def _calibration_array(cfg, train_loader):
    """N_CALIBRATION unaugmented training images, NHWC, already normalized exactly as the
    trainer normalizes them — evenly spaced through the split so every class is covered."""
    loader = build_bn_calibration_loader(cfg.dataset, train_loader)
    ds = loader.dataset
    idx = np.linspace(0, len(ds) - 1, num=min(N_CALIBRATION, len(ds))).round().astype(int)
    x = torch.stack([ds[i][0] for i in idx])
    return x.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)


def _make_interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # pragma: no cover
        from tensorflow.lite import Interpreter
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp


def _tflite_eval(interp, loader):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    logits, labels = [], []
    for x, y in loader:
        for xi, yi in zip(x.permute(0, 2, 3, 1).numpy(), y.numpy()):
            q = np.clip(np.round(xi / in_scale + in_zp), -128, 127).astype(np.int8)[None]
            interp.set_tensor(inp["index"], q)
            interp.invoke()
            o = interp.get_tensor(out["index"]).astype(np.float32)
            logits.append((o - out_zp) * out_scale)
            labels.append(yi)
    return np.concatenate(logits), np.array(labels)


def export_run(run_id, keep_intermediates=False):
    t0 = time.time()
    dataset, student, method, seed = _parse_run_id(run_id)
    ref_path = os.path.join(REPO_ROOT, "results", run_id, "metrics.json")
    ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "students", f"{run_id}.pt")
    with open(ref_path) as f:
        ref = json.load(f)
    out_dir = os.path.join(EXPORT_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    cfg = _compose(dataset, student)
    torch.manual_seed(0)

    # 1. checkpoint purity + strict reload
    state = torch.load(ckpt_path, map_location="cpu")
    purity_check.check_state_dict(state)
    model = build_student_from_cfg(cfg.student, num_classes=cfg.dataset.num_classes)
    model.load_state_dict(state, strict=True)
    model.eval()
    purity_check.check_torch_modules(model)

    # 2. fp32 reload parity against the immutable training record
    train_loader, _, test_loader = _LOADERS[dataset](cfg.dataset)
    torch_logits, labels = _torch_eval(model, test_loader)
    fp32 = _metrics(labels, torch_logits.argmax(1))
    drift = abs(fp32["macro_f1"] - ref["macro_f1"])
    if drift > FP32_TOLERANCE:
        raise purity_check.PurityError(
            f"fp32 reload macro-F1 {fp32['macro_f1']:.6f} != recorded {ref['macro_f1']:.6f} — "
            "export would not be evaluating the trained model")

    # 3. ONNX + parity
    # Static batch 1 (the edge benchmark is batch 1 anyway). A dynamic batch axis makes
    # onnx2tf lower Flatten through a FILL op, which strict full-integer quantization rejects.
    onnx_path = os.path.join(out_dir, "model.onnx")
    dummy = torch.zeros(1, 3, cfg.dataset.image_size, cfg.dataset.image_size)
    torch.onnx.export(model, (dummy,), onnx_path, opset_version=ONNX_OPSET, dynamo=False,
                      input_names=["input"], output_names=["logits"])
    import onnx
    import onnxruntime as ort
    onnx_ops = purity_check.check_onnx(onnx.load(onnx_path))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_logits = np.concatenate([sess.run(None, {"input": xi[None].numpy()})[0]
                                 for x, _ in test_loader for xi in x])
    onnx_parity = {
        "max_abs_logit_diff": float(np.abs(ort_logits - torch_logits).max()),
        "argmax_agreement": float((ort_logits.argmax(1) == torch_logits.argmax(1)).mean()),
    }
    if onnx_parity["argmax_agreement"] < 1.0 or onnx_parity["max_abs_logit_diff"] > 1e-3:
        raise purity_check.PurityError(f"ONNX parity failed: {onnx_parity}")

    # 4. fully-integer INT8 TFLite, per-channel, calibrated on training images
    calib_path = os.path.join(out_dir, "calibration_nhwc.npy")
    np.save(calib_path, _calibration_array(cfg, train_loader))
    tf_dir = os.path.join(out_dir, "tf")
    import onnx2tf
    onnx2tf.convert(
        input_onnx_file_path=onnx_path, output_folder_path=tf_dir,
        output_integer_quantized_tflite=True, quant_type="per-channel",
        # onnx2tf 2.x's default "flatbuffer_direct" writer emits invalid per-channel metadata
        # (quantized_dimension=3 on lower-rank tensors) and fails its own validation; route
        # through TensorFlow's standard TFLite converter instead.
        tflite_backend="tf_converter",
        # calibration data is already normalized exactly as at train/test time -> identity here
        custom_input_op_name_np_data_path=[["input", calib_path, [[[[0.0, 0.0, 0.0]]]], [[[[1.0, 1.0, 1.0]]]]]],
        input_quant_dtype="int8", output_quant_dtype="int8", non_verbose=True,
    )
    produced = glob.glob(os.path.join(tf_dir, "*_full_integer_quant.tflite"))
    if len(produced) != 1:
        raise purity_check.PurityError(f"expected one full-integer TFLite file, found {produced}")
    tflite_path = os.path.join(out_dir, "model_int8.tflite")
    shutil.copyfile(produced[0], tflite_path)
    interp = _make_interpreter(tflite_path)
    tflite_summary = purity_check.check_tflite(interp)

    # 5. INT8 accuracy, identical test split and preprocessing
    int8_logits, int8_labels = _tflite_eval(interp, test_loader)
    assert (int8_labels == labels).all()
    int8 = _metrics(int8_labels, int8_logits.argmax(1))

    if not keep_intermediates:
        shutil.rmtree(tf_dir, ignore_errors=True)
        os.remove(calib_path)

    report = {
        "run_id": run_id, "dataset": dataset, "student": student, "method": method, "seed": seed,
        "recorded": {"macro_f1": ref["macro_f1"], "top1_accuracy": ref["top1_accuracy"]},
        "fp32": fp32,
        "int8": int8,
        "int8_minus_fp32": {k: int8[k] - fp32[k] for k in int8},
        "int8_fp32_argmax_agreement": float((int8_logits.argmax(1) == torch_logits.argmax(1)).mean()),
        "onnx": {"opset": ONNX_OPSET, "ops": onnx_ops, **onnx_parity},
        "tflite": {**tflite_summary, "size_kb": os.path.getsize(tflite_path) / 1024.0,
                   "n_calibration": N_CALIBRATION},
        "params": sum(p.numel() for p in model.parameters()),
        "seconds": round(time.time() - t0, 1),
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnx2tf": onnx2tf.__version__},
    }
    with open(os.path.join(out_dir, "deploy.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info("[%s] exported: fp32 F1 %.4f -> int8 F1 %.4f (%+.4f), %.1f KB, %d convs per-channel",
                run_id, fp32["macro_f1"], int8["macro_f1"], int8["macro_f1"] - fp32["macro_f1"],
                report["tflite"]["size_kb"], tflite_summary["n_conv"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run_id, e.g. trashnet__mobilenetv2_035__kd__s0")
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args()
    report = export_run(args.run, keep_intermediates=args.keep_intermediates)
    print(json.dumps({k: report[k] for k in ("run_id", "fp32", "int8", "int8_minus_fp32", "tflite")}, indent=2))


if __name__ == "__main__":
    main()
