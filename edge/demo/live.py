"""Live webcam demo: classify every frame with the TrashNet and/or Rock-Paper-Scissors student
and print the detections continuously. Ctrl+C to stop.

    ~/qakd_edge/demo                                   # both datasets, strongest models
    ~/qakd_edge/demo --dataset rps_25                  # one dataset only
    ~/qakd_edge/demo --method pqk --student mobilenetv2_035 --seed 0
    ~/qakd_edge/demo --hz 5                            # 5 predictions per second (default 1; 0 = every frame)
    ~/qakd_edge/demo --threshold 0.5                   # say "undefined" less often
    ~/qakd_edge/demo --verbose                         # add confidence, runner-up, latency, FPS

Default models (DEFAULT_MODEL): the strongest exported student per dataset by validation —
mobilenetv3_small + rbf_control on both — INT8, from the bundle built by scripts/edge_bundle.py.
When the top-class probability is below the dataset's THRESHOLD the prediction is "undefined".
Preprocessing matches training: TrashNet photos are 4:3 like the webcam,
so the whole frame is resized (as Resize((128,128)) did); RPS images are square, so the frame is
centre-cropped to a square first instead of stretching the hand. Frames are read continuously
(so each prediction uses a fresh frame) but classified only --hz times per second.

These students were trained on studio photos — TrashNet: one object on a white background; RPS:
rendered hands seen from the back, fingers up, wrist at the bottom, on pure white — hold your hand
that way, filling the frame centre, or it falls back to "rock". Point the camera at a single
object or hand against a plain light background. Neither model has a "nothing there" class, so the
threshold is the only guard: it removes low-confidence guesses, but a scene that resembles a class
(a brown wall looks like cardboard) can still be named confidently.
"""
import argparse
import json
import os
import time
from collections import Counter

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# validation-selected PQK variant per pair (scripts/dashboard.py PQK_CONFIG)
PQK_REPORTED = {("rps_25", "mobilenetv2_035"): "pqk_adaptive", ("trashnet", "mobilenetv3_small"): "pqk_adaptive"}
SQUARE_SOURCE = {"trashnet": False, "rps_25": True}
LABEL = {"trashnet": "TRASH", "rps_25": "RPS"}
# Strongest exported model per dataset, by validation (CLAUDE.md: validation decides): the
# student x method with the highest mean validation macro-F1 over 3 seeds, then its best-validation
# seed. Both datasets pick mobilenetv3_small + rbf_control — trashnet s1: val 0.766, test 0.758;
# rps_25 s0: val 0.970, test 0.959 (INT8 on the Pi: 0.765 / 0.861).
DEFAULT_MODEL = {"trashnet": ("mobilenetv3_small", "rbf_control", 1),
                 "rps_25": ("mobilenetv3_small", "rbf_control", 0)}
# Below this top-class probability the prediction is "undefined". Chosen on validation with the
# default models: trashnet 0.70 keeps 63% of validation images at 90% accuracy (test: 64% at 89%);
# rps_25 0.90 keeps 88% at 100% (test: 79% at 97%), and turned every frame of an empty-desk webcam
# scene undefined (no-hand confidence 0.54-0.85).
THRESHOLD = {"trashnet": 0.70, "rps_25": 0.90}
UNDEFINED = "undefined"


def resolve_run_id(ds, student=None, method=None, seed=None):
    """DEFAULT_MODEL for `ds`, with any of student / method / seed overridden. "pqk" resolves to the
    validation-selected PQK variant for that pair."""
    d_student, d_method, d_seed = DEFAULT_MODEL[ds]
    student, method = student or d_student, method or d_method
    if method == "pqk":
        method = PQK_REPORTED.get((ds, student), "pqk")
    return f"{ds}__{student}__{method}__s{d_seed if seed is None else seed}"


def decide(classes, probs, threshold):
    """(shown label, top class, its probability) — the label is UNDEFINED below the threshold."""
    k = int(np.argmax(probs))
    conf = float(probs[k])
    return (classes[k] if conf >= threshold else UNDEFINED), classes[k], conf


class Student:
    def __init__(self, path, classes, threads):
        self.interp = Interpreter(model_path=path, num_threads=threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]
        self.size = (int(self.inp["shape"][2]), int(self.inp["shape"][1]))
        self.classes = classes

    def predict(self, rgb):
        x = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA).astype(np.float32) / np.float32(255.0)
        x = (x - MEAN) / STD
        q = np.clip(np.round(x / self.in_scale + self.in_zp), -128, 127).astype(np.int8)[None]
        self.interp.set_tensor(self.inp["index"], q)
        t = time.perf_counter()
        self.interp.invoke()
        ms = (time.perf_counter() - t) * 1e3
        logits = (self.interp.get_tensor(self.out["index"]).astype(np.float32)[0] - self.out_zp) * self.out_scale
        e = np.exp(logits - logits.max())
        return e / e.sum(), ms


def centre_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default="~/qakd_edge/bundle")
    p.add_argument("--dataset", default="both", choices=["both", "trashnet", "rps_25"])
    p.add_argument("--student", choices=["mobilenetv2_035", "mobilenetv3_small"], help="default: DEFAULT_MODEL")
    p.add_argument("--method", help="default: DEFAULT_MODEL; pqk (validation-selected variant), pqk_adaptive, kd, rkd, "
                                    "rbf_control, scratch")
    p.add_argument("--seed", type=int, choices=[0, 1, 2], help="default: DEFAULT_MODEL")
    p.add_argument("--threshold", type=float, help="below this probability show 'undefined' (default: per dataset, "
                                                   "trashnet 0.70, rps_25 0.90)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threads", type=int, default=2, help="TFLite threads per model (2 measured best with the camera)")
    p.add_argument("--hz", type=float, default=1.0, help="predictions (and lines) per second; 0 = every frame")
    p.add_argument("--verbose", action="store_true", help="also print confidence, runner-up, latency and FPS")
    p.add_argument("--frames", type=int, default=0, help="stop after N predictions (0 = run until Ctrl+C)")
    args = p.parse_args()
    cv2.setNumThreads(2)

    bundle = os.path.expanduser(args.bundle)
    with open(os.path.join(bundle, "manifest.json")) as f:
        manifest = json.load(f)
    available = {m["id"]: m for m in manifest["models"] if m["precision"] == "int8"}
    datasets = ["trashnet", "rps_25"] if args.dataset == "both" else [args.dataset]
    models, threshold = {}, {}
    for ds in datasets:
        run_id = resolve_run_id(ds, args.student, args.method, args.seed)
        if run_id not in available:
            student = run_id.split("__")[1]
            raise SystemExit(f"no exported model {run_id} in {bundle} — methods for {student}: "
                             + ", ".join(sorted({m['method'] for m in available.values()
                                                 if m['dataset'] == ds and m['student'] == student})))
        models[ds] = Student(os.path.join(bundle, available[run_id]["path"]),
                             manifest["datasets"][ds]["class_names"], args.threads)
        threshold[ds] = THRESHOLD[ds] if args.threshold is None else args.threshold
        print(f"loaded {LABEL[ds]:5s} {run_id}  ({available[run_id]['size_kb']:.0f} KB INT8, "
              f"undefined below {threshold[ds]:.0%})", flush=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")
    for _ in range(15):  # let auto-exposure settle
        cap.read()
    print(f"camera {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} — "
          "running, Ctrl+C to stop\n", flush=True)

    counts = {ds: Counter() for ds in models}
    n, t_start, last_print, fps, prev = 0, time.time(), 0.0, 0.0, time.perf_counter()
    try:
        while not args.frames or n < args.frames:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed — retrying", flush=True)
                time.sleep(0.2)
                continue
            if args.hz > 0 and time.time() - last_print < 1.0 / args.hz:
                continue  # keep the camera buffer fresh; classify only when due
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            parts, infer = [], []
            for ds, model in models.items():
                probs, ms = model.predict(centre_square(rgb) if SQUARE_SOURCE[ds] else rgb)
                label, top, conf = decide(model.classes, probs, threshold[ds])
                counts[ds][label] += 1
                if args.verbose:
                    second = int(np.argsort(probs)[-2])
                    shown = f"{top:<9s} {conf:4.0%}" if label != UNDEFINED else f"{UNDEFINED} ({top} {conf:.0%})"
                    parts.append(f"{LABEL[ds]}: {shown:<26s} 2nd {model.classes[second]:<9s} {probs[second]:4.0%}")
                else:
                    parts.append(f"{LABEL[ds]}: {label:<9s}")
                infer.append(ms)
            n += 1
            now = time.perf_counter()
            inst = 1.0 / max(now - prev, 1e-6)
            fps = inst if n <= 2 else 0.9 * fps + 0.1 * inst  # n=1 interval includes start-up
            prev = now
            if args.hz <= 0 or time.time() - last_print >= 1.0 / args.hz:
                last_print = time.time()
                line = f"{time.strftime('%H:%M:%S')}  " + "  |  ".join(parts)
                if args.verbose:
                    line += f"  |  infer {'+'.join(f'{v:.1f}' for v in infer)} ms  {fps:4.1f} FPS"
                print(line, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
    dur = time.time() - t_start
    print(f"\nstopped: {n} predictions in {dur:.1f} s")
    for ds, c in counts.items():
        print(f"  {LABEL[ds]:5s} " + ", ".join(f"{k} {v}" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
