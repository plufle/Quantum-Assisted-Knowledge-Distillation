"""Live webcam window: show the camera feed with the TrashNet and/or Rock-Paper-Scissors student's
top prediction drawn on it (PC version of live.py, which is the headless Pi demo).

    python edge/demo/live_feed.py                          # both datasets, strongest models, INT8
    python edge/demo/live_feed.py --dataset rps_25         # one dataset only
    python edge/demo/live_feed.py --method pqk --student mobilenetv2_035 --seed 0
    python edge/demo/live_feed.py --backend onnx           # fp32 ONNX Runtime — much faster on x86
    python edge/demo/live_feed.py --rate 5                 # 5 predictions per second (default 1)
    python edge/demo/live_feed.py --threshold 0.5          # say "undefined" less often
q / Esc (or closing the window) quits.

Same models, thresholds, class names and preprocessing as live.py: by default the strongest
student per dataset by validation (mobilenetv3_small + rbf_control), "undefined" below the
per-dataset probability threshold (0.70 for both — see live.THRESHOLD). INT8 TFLite
from the bundle built by scripts/edge_bundle.py, or with --backend onnx the fp32 ONNX export of the
same run (no INT8 rounding, so it can disagree with the deployed model on borderline frames). The video
is drawn at the camera's frame rate while a background thread classifies the most recent frame
once per second (--rate), so labels do not flicker and a slow model never stalls the feed — on
Windows, LiteRT has no fast INT8 kernels (~18 ms per model at 8 threads vs ~2.5 ms on the Pi 4). For RPS the box drawn in the middle is the
square crop the model sees. RPS was trained on rendered hands seen from the back, fingers up,
wrist at the bottom, on pure white — hold your hand that way, filling the box, against a white
wall, or it falls back to "rock". Neither model has a "nothing there" class.
"""
import argparse
import json
import os
import threading
import time

import cv2
import numpy as np

from live import (LABEL, MEAN, SQUARE_SOURCE, STD, THRESHOLD, UNDEFINED, Student, centre_square, decide,
                  resolve_run_id)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WINDOW = "QAKD live - q to quit"


class OnnxStudent:
    """fp32 ONNX Runtime twin of live.Student (same preprocessing, same predict signature)."""

    def __init__(self, path, classes, threads):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape  # [1, 3, H, W]
        self.size = (int(shape[3]), int(shape[2]))
        self.classes = classes

    def predict(self, rgb):
        x = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA).astype(np.float32) / np.float32(255.0)
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        t = time.perf_counter()
        logits = self.sess.run(None, {self.name: np.ascontiguousarray(x)})[0][0]
        ms = (time.perf_counter() - t) * 1e3
        e = np.exp(logits - logits.max())
        return e / e.sum(), ms


class TeacherModel:
    """The ResNet-50 teacher for a dataset (ImageNet-pretrained, fine-tuned on it), fp32 PyTorch on
    the CPU — shows whether pretraining alone travels better to live scenes than the from-scratch
    students. Same preprocessing and predict signature as live.Student."""

    def __init__(self, ds, classes, threads, ckpt=None):
        import torch
        from qakd.models.teachers.resnet import ResNet50Teacher
        torch.set_num_threads(threads)
        self.torch = torch
        self.model = ResNet50Teacher(num_classes=len(classes), pretrained=False)
        ckpt = ckpt or os.path.join(REPO, "checkpoints", "teachers", f"resnet50_in1k__{ds}.pt")
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()
        self.size = (128, 128)  # the teachers were fine-tuned at the datasets' 128x128
        self.classes = classes

    def predict(self, rgb):
        x = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA).astype(np.float32) / np.float32(255.0)
        x = self.torch.from_numpy(np.ascontiguousarray(((x - MEAN) / STD).transpose(2, 0, 1)[None]))
        t = time.perf_counter()
        with self.torch.no_grad():
            logits = self.model(x)[0].numpy()
        ms = (time.perf_counter() - t) * 1e3
        e = np.exp(logits - logits.max())
        return e / e.sum(), ms


class Worker(threading.Thread):
    """Classifies the most recent frame at most `rate` times per second (0 = as fast as it can);
    frames in between are skipped."""

    def __init__(self, models, thresholds, rate):
        super().__init__(daemon=True)
        self.models, self.thresholds = models, thresholds
        self.lock, self.ready = threading.Lock(), threading.Event()
        self.frame, self.result, self.probs, self.stop = None, {}, {}, False
        self.infer_ms, self.rate, self.seq = {}, 0.0, 0
        self.period = 1.0 / rate if rate > 0 else 0.0

    def submit(self, rgb):
        with self.lock:
            self.frame = rgb
        self.ready.set()

    def run(self):
        prev, due = time.perf_counter(), 0.0
        while not self.stop:
            if not self.ready.wait(0.1):
                continue
            wait = due - time.perf_counter()
            if wait > 0:  # not due yet; keep the newest frame and check again
                time.sleep(min(wait, 0.05))
                continue
            due = time.perf_counter() + self.period
            with self.lock:
                rgb, self.frame = self.frame, None
                self.ready.clear()
            if rgb is None:
                continue
            result, all_probs, infer = {}, {}, {}
            for ds, model in self.models.items():
                probs, ms = model.predict(centre_square(rgb) if SQUARE_SOURCE[ds] else rgb)
                result[ds] = decide(model.classes, probs, self.thresholds[ds])[0]
                all_probs[ds] = (model.classes, probs)
                infer[ds] = ms
            now = time.perf_counter()
            self.rate = 0.9 * self.rate + 0.1 / max(now - prev, 1e-6) if self.rate else 1.0 / max(now - prev, 1e-6)
            prev = now
            self.result, self.probs, self.infer_ms = result, all_probs, infer
            self.seq += 1


def draw(frame, result, probs, thresholds, infer_ms, cam_fps, pred_rate, show_box, mirror):
    """Video frame with a side panel per dataset: the decided label, every class's probability as a
    bar, and the "undefined" threshold as an orange line across the bars."""
    h, w = frame.shape[:2]
    if show_box:  # the square the RPS model sees
        s = min(h, w)
        x0, y0 = (w - s) // 2, (h - s) // 2
        cv2.rectangle(frame, (x0, y0), (x0 + s - 1, y0 + s - 1), (255, 255, 255), 1, cv2.LINE_AA)
        if not mirror:
            cv2.putText(frame, "RPS: back of hand, fingers up, white background", (x0 + 6, y0 + s - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if mirror:
        frame = cv2.flip(frame, 1)
    font, white, orange, green, grey = cv2.FONT_HERSHEY_SIMPLEX, (235, 235, 235), (0, 165, 255), (80, 255, 120), (110, 110, 110)
    shown = [ds for ds in result if ds in probs]
    panel_w, bar_x, bar_w, row = 345, 108, 160, 22
    panel_h = 12 + sum(36 + row * len(probs[ds][0]) + 22 for ds in shown) + 26 if shown else 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, min(panel_h, h)), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    y = 8
    if not shown:
        cv2.putText(frame, "warming up...", (12, 40), font, 0.9, white, 2, cv2.LINE_AA)
    for ds in shown:
        classes, p = probs[ds]
        label, thr = result[ds], thresholds[ds]
        y += 32
        cv2.putText(frame, f"{LABEL[ds]}: {label}", (12, y), font, 0.95, orange if label == UNDEFINED else green, 2,
                    cv2.LINE_AA)
        top = int(np.argmax(p))
        bars_top = y + 6
        for k, (cls, v) in enumerate(zip(classes, p)):
            y += row
            cv2.putText(frame, cls, (14, y), font, 0.5, white, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (bar_x, y - 12), (bar_x + bar_w, y - 1), (60, 60, 60), -1)
            fill = green if (k == top and v >= thr) else grey
            cv2.rectangle(frame, (bar_x, y - 12), (bar_x + int(bar_w * float(v)), y - 1), fill, -1)
            cv2.putText(frame, f"{float(v):4.0%}", (bar_x + bar_w + 8, y - 1), font, 0.5, white, 1, cv2.LINE_AA)
        tx = bar_x + int(bar_w * thr)  # threshold marker across this dataset's bars
        cv2.line(frame, (tx, bars_top), (tx, y + 2), orange, 2)
        y += 18
        cv2.putText(frame, f"threshold {thr:.0%} (orange line)", (bar_x, y), font, 0.42, orange, 1, cv2.LINE_AA)
        y += 4
    stats = (f"infer {'+'.join(f'{v:.0f}' for v in infer_ms.values())} ms | "
             f"video {cam_fps:.0f} FPS | {pred_rate:.1f} pred/s")
    cv2.putText(frame, stats, (12, min(panel_h, h) - 8), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default=os.path.join(REPO, "exports", "_edge_bundle"))
    p.add_argument("--dataset", default="both", choices=["both", "trashnet", "rps_25"])
    p.add_argument("--student", choices=["mobilenetv2_035", "mobilenetv3_small"], help="default: live.DEFAULT_MODEL")
    p.add_argument("--method", help="default: live.DEFAULT_MODEL; pqk (validation-selected variant), pqk_adaptive, kd, "
                                    "rkd, rbf_control, scratch")
    p.add_argument("--seed", type=int, choices=[0, 1, 2], help="default: live.DEFAULT_MODEL")
    p.add_argument("--threshold", type=float, help="below this probability show 'undefined' (default: per dataset, "
                                                   "0.70 for both datasets)")
    p.add_argument("--backend", default="int8", choices=["int8", "onnx"],
                   help="int8 = the deployed TFLite model; onnx = fp32 export of the same run")
    p.add_argument("--realworld", choices=["teacher", "kd", "pqk_adaptive"],
                   help="the separate real-world RPS model (realworld/): its teacher (fp32) or an INT8 student; "
                        "4 classes incl. 'none', whole-frame input")
    p.add_argument("--teacher", action="store_true",
                   help="run the ResNet-50 teachers (fp32 PyTorch) instead of the INT8 students")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--mirror", action="store_true", help="mirror the displayed video (models see the unmirrored frame)")
    p.add_argument("--rate", type=float, default=1.0, help="predictions per second; 0 = as fast as possible")
    p.add_argument("--quiet", action="store_true", help="no console output")
    args = p.parse_args()

    with open(os.path.join(args.bundle, "manifest.json")) as f:
        manifest = json.load(f)
    available = {m["id"]: m for m in manifest["models"] if m["precision"] == "int8"}
    datasets = ["trashnet", "rps_25"] if args.dataset == "both" else [args.dataset]
    models, thresholds = {}, {}
    if args.realworld:
        datasets = []
        rw_classes = ["paper", "rock", "scissors", "none"]
        if args.realworld == "teacher":
            models["rps_real"] = TeacherModel("rps_real", rw_classes, args.threads,
                                              os.path.join(REPO, "realworld", "runs", "teacher_resnet50", "best.pt"))
        else:
            path = os.path.join(REPO, "realworld", "exports", f"student_mnv3s_{args.realworld}_s0", "model_drq.tflite")
            models["rps_real"] = Student(path, rw_classes, args.threads)
        thresholds["rps_real"] = 0.70 if args.threshold is None else args.threshold
        print(f"loaded RPS   real-world {args.realworld} (undefined below {thresholds['rps_real']:.0%})", flush=True)
    for ds in datasets:
        if args.teacher:
            models[ds] = TeacherModel(ds, manifest["datasets"][ds]["class_names"], args.threads)
            thresholds[ds] = THRESHOLD[ds] if args.threshold is None else args.threshold
            print(f"loaded {LABEL[ds]:5s} teacher resnet50_in1k__{ds} (fp32 PyTorch, undefined below "
                  f"{thresholds[ds]:.0%})", flush=True)
            continue
        run_id = resolve_run_id(ds, args.student, args.method, args.seed)
        if run_id not in available:
            student = run_id.split("__")[1]
            raise SystemExit(f"no exported model {run_id} — methods for {student}: " + ", ".join(sorted(
                {m["method"] for m in available.values() if m["dataset"] == ds and m["student"] == student})))
        classes = manifest["datasets"][ds]["class_names"]
        if args.backend == "onnx":
            models[ds] = OnnxStudent(os.path.join(REPO, "exports", run_id, "model.onnx"), classes, args.threads)
        else:
            models[ds] = Student(os.path.join(args.bundle, available[run_id]["path"]), classes, args.threads)
        thresholds[ds] = THRESHOLD[ds] if args.threshold is None else args.threshold
        print(f"loaded {LABEL[ds]:5s} {run_id} ({args.backend}, undefined below {thresholds[ds]:.0%})", flush=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # DirectShow opens in ~2 s, MSMF in ~15 s
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")
    worker = Worker(models, thresholds, args.rate)
    worker.start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("running — q / Esc in the video window to quit", flush=True)

    cam_fps, prev, last_seq = 0.0, time.perf_counter(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed — retrying", flush=True)
                time.sleep(0.2)
                continue
            worker.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            now = time.perf_counter()
            inst = 1.0 / max(now - prev, 1e-6)
            cam_fps = 0.9 * cam_fps + 0.1 * inst if cam_fps else inst
            prev = now
            result, probs = dict(worker.result), dict(worker.probs)
            cv2.imshow(WINDOW, draw(frame, result, probs, thresholds, dict(worker.infer_ms), cam_fps, worker.rate,
                                    "rps_25" in models, args.mirror))
            if not args.quiet and result and worker.seq != last_seq:  # each new prediction once
                last_seq = worker.seq
                print(f"{time.strftime('%H:%M:%S')}  " + "  |  ".join(
                    f"{LABEL[d]}: {c:<9s} (" + " ".join(f"{n} {float(v):.0%}" for n, v in zip(*probs[d])) + ")"
                    for d, c in result.items() if d in probs), flush=True)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop = True
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
