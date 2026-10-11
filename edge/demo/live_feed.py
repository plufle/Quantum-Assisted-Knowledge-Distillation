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
per-dataset probability threshold (trashnet 0.70, rps_25 0.90 — see live.THRESHOLD). INT8 TFLite
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


class Worker(threading.Thread):
    """Classifies the most recent frame at most `rate` times per second (0 = as fast as it can);
    frames in between are skipped."""

    def __init__(self, models, thresholds, rate):
        super().__init__(daemon=True)
        self.models, self.thresholds = models, thresholds
        self.lock, self.ready = threading.Lock(), threading.Event()
        self.frame, self.result, self.stop = None, {}, False
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
            result, infer = {}, {}
            for ds, model in self.models.items():
                probs, ms = model.predict(centre_square(rgb) if SQUARE_SOURCE[ds] else rgb)
                result[ds] = decide(model.classes, probs, self.thresholds[ds])[0]
                infer[ds] = ms
            now = time.perf_counter()
            self.rate = 0.9 * self.rate + 0.1 / max(now - prev, 1e-6) if self.rate else 1.0 / max(now - prev, 1e-6)
            prev = now
            self.result, self.infer_ms = result, infer
            self.seq += 1


def draw(frame, result, infer_ms, cam_fps, pred_rate, show_box, mirror):
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
    lines = [(f"{LABEL[ds]}: {cls}", (0, 165, 255) if cls == UNDEFINED else (80, 255, 120))  # BGR: orange / green
             for ds, cls in result.items()] or [("warming up...", (220, 220, 220))]
    stats = (f"infer {'+'.join(f'{v:.0f}' for v in infer_ms.values())} ms | "
             f"video {cam_fps:.0f} FPS | predictions {pred_rate:.1f}/s")
    box_h = 18 + 40 * len(lines) + 26
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, box_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    for k, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (14, 44 + 40 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2, cv2.LINE_AA)
    cv2.putText(frame, stats, (14, box_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
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
                                                   "trashnet 0.70, rps_25 0.90)")
    p.add_argument("--backend", default="int8", choices=["int8", "onnx"],
                   help="int8 = the deployed TFLite model; onnx = fp32 export of the same run")
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
    for ds in datasets:
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
            result = dict(worker.result)
            cv2.imshow(WINDOW, draw(frame, result, dict(worker.infer_ms), cam_fps, worker.rate,
                                    "rps_25" in models, args.mirror))
            if not args.quiet and result and worker.seq != last_seq:  # each new prediction once
                last_seq = worker.seq
                print(f"{time.strftime('%H:%M:%S')}  " + "  |  ".join(f"{LABEL[d]}: {c:<9s}" for d, c in result.items()),
                      flush=True)
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
