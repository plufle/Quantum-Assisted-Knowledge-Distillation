"""Live camera classification — end-to-end latency of the deployed pipeline on the device.

    python edge/bench/camera.py --bundle ~/qakd_edge/bundle --id trashnet__mobilenetv2_035__kd__s0

Timed per frame:
  capture     V4L2 read + MJPEG decode (blocks on the camera's frame interval)
  preprocess  BGR->RGB, resize to the model input, ImageNet normalize, int8 quantize
  invoke      TFLite inference
  postprocess dequantize, softmax, argmax
Reports per-stage and end-to-end latency, achieved FPS, compute-bound FPS (without capture),
process CPU utilisation, SoC temperature / clock, RSS, and saves a few annotated frames.

Preprocessing matches training except for the resampler: torchvision Resize on PIL uses
antialiased bilinear; cv2.INTER_AREA is the closest OpenCV equivalent for downscaling.
Live frames are out of distribution (TrashNet is single objects on a white background), so
predictions here are a demonstration, not an accuracy measurement — run.py measures accuracy.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from run import Sampler, _make_io, _read, FREQ, mem_kb, proc_stat, stats, temp_c, throttled, util_pct

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def open_camera(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {index}")
    return cap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True)
    p.add_argument("--id", required=True, help="model id from manifest.json")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cv-threads", type=int, default=None,
                   help="cv2.setNumThreads; OpenCV's pool competes with the TFLite pool for cores")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--save", type=int, default=6, help="annotated frames to save, evenly spaced")
    p.add_argument("--out", default="results/camera")
    args = p.parse_args()

    if args.cv_threads is not None:
        cv2.setNumThreads(args.cv_threads)
    bundle = os.path.expanduser(args.bundle)
    with open(os.path.join(bundle, "manifest.json")) as f:
        manifest = json.load(f)
    m = next(m for m in manifest["models"] if m["id"] == args.id)
    classes = manifest["datasets"][m["dataset"]]["class_names"]
    tag = f"t{args.threads}_cv{'default' if args.cv_threads is None else args.cv_threads}"
    out_dir = os.path.join(args.out, f"{args.id}__{tag}")
    os.makedirs(out_dir, exist_ok=True)

    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=os.path.join(bundle, m["path"]), num_threads=args.threads)
    interp.allocate_tensors()
    inp, out, prep, post = _make_io(interp)
    size = (int(inp["shape"][2]), int(inp["shape"][1]))  # cv2 wants (w, h)

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              float(cap.get(cv2.CAP_PROP_FPS)))

    def frame_once():
        t0 = time.perf_counter_ns()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        t1 = time.perf_counter_ns()
        rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA)
        x = (rgb.astype(np.float32) / np.float32(255.0) - MEAN) / STD
        interp.set_tensor(inp["index"], prep(x))
        t2 = time.perf_counter_ns()
        interp.invoke()
        t3 = time.perf_counter_ns()
        logits = post(interp.get_tensor(out["index"]))[0]
        e = np.exp(logits - logits.max())
        prob = e / e.sum()
        k = int(prob.argmax())
        t4 = time.perf_counter_ns()
        return frame, k, float(prob[k]), [(t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6, (t4 - t3) / 1e6]

    for _ in range(args.warmup):  # lets auto-exposure settle and the clock ramp up
        frame_once()

    save_at = set(np.linspace(0, args.frames - 1, num=min(args.save, args.frames)).round().astype(int).tolist())
    max_mhz = int(_read(FREQ.format(0, "scaling_max_freq")) or 0) / 1000.0
    s0, c0, th0 = proc_stat(), os.times(), throttled()
    sampler = Sampler(0.1)
    sampler.start()
    stages, frame_ms, preds = [], [], []
    w0 = time.perf_counter_ns()
    for i in range(args.frames):
        f0 = time.perf_counter_ns()
        frame, k, conf, st = frame_once()
        frame_ms.append((time.perf_counter_ns() - f0) / 1e6)
        stages.append(st)
        preds.append(k)
        if i in save_at:
            label = f"{classes[k]} {conf:.2f}  |  infer {st[2]:.1f} ms"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.jpg"), frame)
    wall_s = (time.perf_counter_ns() - w0) / 1e9
    sampler.stop()
    s1, c1 = proc_stat(), os.times()
    cap.release()

    st = np.array(stages)
    processing = st[:, 1:].sum(1)
    cpu_s = (c1.user - c0.user) + (c1.system - c0.system)
    report = {
        "id": args.id, "dataset": m["dataset"], "student": m["student"], "method": m["method"],
        "threads": args.threads, "cv_threads": cv2.getNumThreads(), "config": tag, "camera": {"requested": [args.width, args.height, args.fps],
                                            "actual": list(actual)},
        "frames": args.frames, "warmup": args.warmup,
        "capture_ms": stats(st[:, 0]), "preprocess_ms": stats(st[:, 1]), "invoke_ms": stats(st[:, 2]),
        "postprocess_ms": stats(st[:, 3]), "processing_ms": stats(processing),
        "frame_ms": stats(frame_ms),
        "achieved_fps": round(args.frames / wall_s, 2),
        "compute_bound_fps": round(1000.0 / float(processing.mean()), 2),
        "cpu": {"process_util_pct": round(100.0 * cpu_s / wall_s, 1),
                "system_util_pct": util_pct(s0, s1)},
        "thermal": {"throttled_before": th0, "throttled_after": throttled(), **sampler.summary(max_mhz)},
        "memory_kb": mem_kb(),
        "predictions": {classes[c]: int(n) for c, n in zip(*np.unique(preds, return_counts=True))},
        "note": "live frames are out of distribution — predictions are a demonstration only",
    }
    with open(os.path.join(out_dir, "camera.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("id", "achieved_fps", "compute_bound_fps", "frame_ms",
                                              "invoke_ms", "preprocess_ms", "cpu", "predictions")}, indent=1))


if __name__ == "__main__":
    main()
