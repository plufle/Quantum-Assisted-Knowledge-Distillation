"""Build the on-device benchmark bundle (CLAUDE.md Stage 10) from the verified exports.

    python scripts/edge_bundle.py

Writes exports/_edge_bundle/ (gitignored, copied to the device as-is):
  data/{dataset}_x.npy, _y.npy  test split, NHWC float32, normalized exactly as at train/test time
  models/int8/{run_id}.tflite   every exported INT8 model (66)
  models/fp32/{arch}.tflite     float32 TFLite of one model per dataset x student — same runtime
                                as INT8, so the fp32-vs-INT8 latency comparison is not a runtime one
  reference/{model_id}.npy      the PC's dequantized logits for every model on its test split —
                                the device must reproduce these (parity is hardware, not software)
  manifest.json                 models, datasets, params, size, MACs, PC metrics

Nothing here retrains or re-exports: INT8 models are copied from exports/, and every PC INT8
score recomputed from the arrays must equal deploy.json exactly, or the bundle is not built.
"""
import glob
import json
import os
import shutil
from collections import Counter

import numpy as np
import torch

from qakd.deploy.export import (EXPORT_ROOT, _LOADERS, _compose, _make_interpreter, _metrics,
                                _tflite_eval)
from qakd.logger import logger

BUNDLE = os.path.join(EXPORT_ROOT, "_edge_bundle")
# one fp32 model per dataset x student; latency is set by architecture, not by method or seed
FP32_METHOD, FP32_SEED = "pqk", 0


def _dump_test_split(dataset):
    cfg = _compose(dataset, "mobilenetv2_035")  # the test split depends on the dataset only
    _, _, test_loader = _LOADERS[dataset](cfg.dataset)
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32))
        ys.append(y.numpy().astype(np.int64))
    x, y = np.concatenate(xs), np.concatenate(ys)
    np.save(os.path.join(BUNDLE, "data", f"{dataset}_x.npy"), x)
    np.save(os.path.join(BUNDLE, "data", f"{dataset}_y.npy"), y)
    return test_loader, {"n_test": int(len(y)), "image_size": int(cfg.dataset.image_size),
                         "class_names": list(cfg.dataset.class_names),
                         "x": f"data/{dataset}_x.npy", "y": f"data/{dataset}_y.npy"}


def _graph_macs(interp):
    """Multiply-accumulates of every CONV_2D / DEPTHWISE_CONV_2D / FULLY_CONNECTED op, read off
    the TFLite graph (elementwise ADD/MEAN/PAD are not counted)."""
    shapes = {t["index"]: t["shape"] for t in interp.get_tensor_details()}
    by_op = Counter()
    for op in interp._get_ops_details():
        name, ins, outs = op["op_name"], op["inputs"], op["outputs"]
        if name == "CONV_2D":        # weight [oc, kh, kw, ic]
            w, o = shapes[ins[1]], shapes[outs[0]]
            by_op[name] += int(np.prod(o[1:])) * int(w[1] * w[2] * w[3])
        elif name == "DEPTHWISE_CONV_2D":  # weight [1, kh, kw, oc]
            w, o = shapes[ins[1]], shapes[outs[0]]
            by_op[name] += int(np.prod(o[1:])) * int(w[1] * w[2])
        elif name == "FULLY_CONNECTED":    # weight [out, in]
            w = shapes[ins[1]]
            by_op[name] += int(w[0] * w[1])
    return int(sum(by_op.values())), dict(by_op)


def _fp32_tflite(run_id, out_path):
    import onnx2tf
    tmp = os.path.join(BUNDLE, "_tmp", run_id)
    shutil.rmtree(tmp, ignore_errors=True)
    onnx2tf.convert(input_onnx_file_path=os.path.join(EXPORT_ROOT, run_id, "model.onnx"),
                    output_folder_path=tmp, tflite_backend="tf_converter", non_verbose=True)
    shutil.copyfile(os.path.join(tmp, "model_float32.tflite"), out_path)
    shutil.rmtree(os.path.join(BUNDLE, "_tmp"), ignore_errors=True)


def _float_eval(interp, x):
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    logits = []
    for xi in x:
        interp.set_tensor(inp["index"], xi[None])
        interp.invoke()
        logits.append(interp.get_tensor(out["index"]).astype(np.float32))
    return np.concatenate(logits)


def main():
    torch.manual_seed(0)
    for sub in ("data", "models/int8", "models/fp32", "reference"):
        os.makedirs(os.path.join(BUNDLE, sub), exist_ok=True)

    runs = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(EXPORT_ROOT, "*", "model_int8.tflite")))
    datasets, loaders = {}, {}
    for ds in sorted({r.split("__")[0] for r in runs}):
        loaders[ds], datasets[ds] = _dump_test_split(ds)
        logger.info("[bundle] %s test split: %d images", ds, datasets[ds]["n_test"])

    arrays = {ds: (np.load(os.path.join(BUNDLE, d["x"])), np.load(os.path.join(BUNDLE, d["y"])))
              for ds, d in datasets.items()}
    models, macs_cache = [], {}
    for run_id in runs:
        with open(os.path.join(EXPORT_ROOT, run_id, "deploy.json")) as f:
            dep = json.load(f)
        dataset, student = dep["dataset"], dep["student"]
        x, y = arrays[dataset]
        src = os.path.join(EXPORT_ROOT, run_id, "model_int8.tflite")
        dst = os.path.join(BUNDLE, "models", "int8", f"{run_id}.tflite")
        ref_path = os.path.join(BUNDLE, "reference", f"{run_id}.npy")
        interp = _make_interpreter(src)
        # resumable: the model and its reference are written only after the parity check passes
        if not (os.path.exists(dst) and os.path.exists(ref_path)):
            # recompute from the dumped arrays with export.py's own routine; must equal deploy.json
            logits, labels = _tflite_eval(interp, [(torch.from_numpy(x).permute(0, 3, 1, 2), torch.from_numpy(y))])
            m = _metrics(labels, logits.argmax(1))
            if m != dep["int8"]:
                raise RuntimeError(f"{run_id}: bundle INT8 {m} != deploy.json {dep['int8']}")
            np.save(ref_path, logits)
            shutil.copyfile(src, dst + ".tmp")
            os.replace(dst + ".tmp", dst)  # atomic: a partial copy never counts as done
            logger.info("[bundle] %s int8 verified (F1 %.4f)", run_id, m["macro_f1"])
        if (dataset, student) not in macs_cache:
            macs_cache[(dataset, student)] = _graph_macs(interp)
        macs, macs_by_op = macs_cache[(dataset, student)]
        models.append({"id": run_id, "path": f"models/int8/{run_id}.tflite", "precision": "int8",
                       "dataset": dataset, "student": student, "method": dep["method"],
                       "seed": dep["seed"], "params": dep["params"],
                       "size_kb": round(os.path.getsize(src) / 1024.0, 2), "macs": macs,
                       "macs_by_op": macs_by_op, "pc_metrics": dep["int8"],
                       "pc_fp32_metrics": dep["fp32"], "reference": f"reference/{run_id}.npy"})

    for dataset, student in sorted(macs_cache):
        run_id = f"{dataset}__{student}__{FP32_METHOD}__s{FP32_SEED}"
        model_id = f"{run_id}__fp32"
        out_path = os.path.join(BUNDLE, "models", "fp32", f"{model_id}.tflite")
        _fp32_tflite(run_id, out_path)
        with open(os.path.join(EXPORT_ROOT, run_id, "deploy.json")) as f:
            dep = json.load(f)
        x = np.load(os.path.join(BUNDLE, datasets[dataset]["x"]))
        y = np.load(os.path.join(BUNDLE, datasets[dataset]["y"]))
        interp = _make_interpreter(out_path)
        logits = _float_eval(interp, x)
        m = _metrics(y, logits.argmax(1))
        # float conversion may flip a borderline argmax; report rather than demand bit-exactness
        logger.info("[bundle] %s fp32 TFLite F1 %.4f (torch fp32 %.4f)", model_id, m["macro_f1"],
                    dep["fp32"]["macro_f1"])
        np.save(os.path.join(BUNDLE, "reference", f"{model_id}.npy"), logits)
        macs, macs_by_op = _graph_macs(interp)
        models.append({"id": model_id, "path": f"models/fp32/{model_id}.tflite", "precision": "fp32",
                       "dataset": dataset, "student": student, "method": FP32_METHOD,
                       "seed": FP32_SEED, "params": dep["params"],
                       "size_kb": round(os.path.getsize(out_path) / 1024.0, 2), "macs": macs,
                       "macs_by_op": macs_by_op, "pc_metrics": m, "pc_fp32_metrics": dep["fp32"],
                       "reference": f"reference/{model_id}.npy"})

    manifest = {"datasets": datasets, "models": models,
                "preprocessing": {"resize": "torchvision Resize((128,128)), PIL bilinear",
                                  "normalize_mean": [0.485, 0.456, 0.406],
                                  "normalize_std": [0.229, 0.224, 0.225], "layout": "NHWC"}}
    with open(os.path.join(BUNDLE, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("[bundle] %d models (%d int8, %d fp32) -> %s", len(models),
                sum(m["precision"] == "int8" for m in models),
                sum(m["precision"] == "fp32" for m in models), BUNDLE)


if __name__ == "__main__":
    main()
