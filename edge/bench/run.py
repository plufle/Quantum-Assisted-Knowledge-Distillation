"""On-device benchmark for exported QAKD students (CLAUDE.md Stage 10).

Single model:
    python edge/bench/run.py --model model_int8.tflite --device rpi4 [--threads 4]
Suite over the bundle built by scripts/edge_bundle.py:
    python edge/bench/run.py --bundle ~/qakd_edge/bundle --device rpi4 --suite all --out results/rpi4

Protocol (CLAUDE.md Deploy): 50 warmup + 500 timed invocations, batch 1, timed per iteration
with perf_counter_ns. Logged for every model:
  - load time (Interpreter + allocate_tensors) and cold first-inference latency
  - invoke latency: mean/std/min/p50/p90/p95/p99/max, and throughput
  - pipeline latency: quantize -> set_tensor -> invoke -> get_tensor -> dequantize -> argmax
    (image decode/resize is not included; edge/bench/camera.py measures the full camera path)
  - process CPU utilisation = (user + sys CPU time) / wall time — 100% = one core busy
  - system per-core utilisation from /proc/stat over the timed window
  - CPU frequency and SoC temperature sampled every 100 ms, firmware throttling flags
  - context switches, page faults, RSS / anonymous RSS / peak RSS at each stage
  - with a test split: on-device macro-F1 / top-1, and parity against the PC's logits
Each model runs in its own subprocess, so memory and thread pools never carry over.
Energy is NOT measured — that needs an INA219 on the 5V rail (CLAUDE.md Deploy).
"""
import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time

import numpy as np

THERMAL = "/sys/class/thermal/thermal_zone0/temp"
FREQ = "/sys/devices/system/cpu/cpu{}/cpufreq/{}"
THROTTLED = "/sys/devices/platform/soc/soc:firmware/get_throttled"
THROTTLE_BITS = {0: "under_voltage", 1: "arm_freq_capped", 2: "throttled", 3: "soft_temp_limit",
                 16: "under_voltage_occurred", 17: "arm_freq_capped_occurred",
                 18: "throttled_occurred", 19: "soft_temp_limit_occurred"}


# ------------------------------------------------------------------ system probes
def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def temp_c():
    v = _read(THERMAL)
    return int(v) / 1000.0 if v else None


def freq_mhz(cpu=0):  # Raspberry Pi 4: all four cores share one clock
    v = _read(FREQ.format(cpu, "scaling_cur_freq"))
    return int(v) / 1000.0 if v else None


def throttled():
    v = _read(THROTTLED)
    if v is None:
        return None
    bits = int(v, 16)
    return {"raw": hex(bits), "flags": [n for b, n in THROTTLE_BITS.items() if bits >> b & 1]}


def proc_stat():
    """{cpu, cpu0..N: (busy_jiffies, total_jiffies)} — idle and iowait count as not busy."""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                break
            name, *vals = line.split()
            vals = [int(v) for v in vals[:8]]  # user nice system idle iowait irq softirq steal
            idle = vals[3] + vals[4]
            out[name] = (sum(vals) - idle, sum(vals))
    return out


def util_pct(a, b):
    return {k: round(100.0 * (b[k][0] - a[k][0]) / max(1, b[k][1] - a[k][1]), 1) for k in b}


def mem_kb():
    d = {}
    with open("/proc/self/status") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("VmRSS", "VmHWM", "RssAnon", "RssFile"):
                d[k] = int(v.split()[0])
    return d


def stats(ms):
    a = np.asarray(ms, dtype=np.float64)
    q = np.percentile(a, [50, 90, 95, 99])
    return {"n": int(a.size), "mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
            "min": round(float(a.min()), 4), "p50": round(float(q[0]), 4), "p90": round(float(q[1]), 4),
            "p95": round(float(q[2]), 4), "p99": round(float(q[3]), 4), "max": round(float(a.max()), 4)}


class Sampler(threading.Thread):
    """Temperature and CPU clock every `interval` seconds while the timed loop runs."""

    def __init__(self, interval):
        super().__init__(daemon=True)
        self.interval, self.done = interval, threading.Event()
        self.temp, self.freq = [], []

    def run(self):
        while not self.done.is_set():
            self.temp.append(temp_c())
            self.freq.append(freq_mhz())
            self.done.wait(self.interval)

    def stop(self):
        self.done.set()
        self.join()

    def summary(self, max_mhz):
        t = [v for v in self.temp if v is not None]
        f = [v for v in self.freq if v is not None]
        out = {"n_samples": len(self.temp)}
        if t:
            out["temp_c"] = {"mean": round(float(np.mean(t)), 2), "max": max(t), "min": min(t)}
        if f:
            out["freq_mhz"] = {"mean": round(float(np.mean(f)), 1), "min": min(f), "max": max(f),
                               "frac_at_max": round(float(np.mean([v >= max_mhz for v in f])), 3)}
        return out


def macro_f1(y, p):
    """sklearn f1_score(average='macro') over the union of true and predicted labels."""
    f1 = []
    for c in np.union1d(y, p):
        tp = int(np.sum((p == c) & (y == c)))
        d = 2 * tp + int(np.sum((p == c) & (y != c))) + int(np.sum((p != c) & (y == c)))
        f1.append(2 * tp / d if d else 0.0)
    return float(np.mean(f1))


# ------------------------------------------------------------------ one model (worker)
def _make_io(interp):
    """Same quantize / dequantize as qakd.deploy.export._tflite_eval, so logits are comparable."""
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    if inp["dtype"] == np.int8:
        def prep(xi):
            return np.clip(np.round(xi / in_scale + in_zp), -128, 127).astype(np.int8)[None]
    else:
        def prep(xi):
            return np.ascontiguousarray(xi, dtype=np.float32)[None]
    if out["dtype"] == np.int8:
        def post(o):
            return (o.astype(np.float32) - out_zp) * out_scale
    else:
        def post(o):
            return o.astype(np.float32)
    return inp, out, prep, post


def bench_model(model, threads=4, warmup=50, runs=500, data_x=None, data_y=None, reference=None,
                accuracy=True, resolver="auto", sample_interval=0.1):
    mem = {"baseline": mem_kb()}
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    mem["after_import"] = mem_kb()

    kwargs = {"model_path": model, "num_threads": threads}
    if resolver == "no_default_delegates":  # builtin kernels only, XNNPACK not applied
        kwargs["experimental_op_resolver_type"] = OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    t0 = time.perf_counter_ns()
    interp = Interpreter(**kwargs)
    interp.allocate_tensors()
    load_ms = (time.perf_counter_ns() - t0) / 1e6
    mem["after_load"] = mem_kb()
    inp, out, prep, post = _make_io(interp)

    if data_x:
        x = np.load(data_x, mmap_mode="r")
        inputs = [np.array(x[i]) for i in range(min(64, len(x)))]
    else:  # no test split: deterministic N(0,1) inputs, the scale of normalized images
        rng = np.random.default_rng(0)
        inputs = [rng.standard_normal(tuple(inp["shape"][1:]), dtype=np.float32) for _ in range(64)]

    def once(xi):
        a = time.perf_counter_ns()
        interp.set_tensor(inp["index"], prep(xi))
        b = time.perf_counter_ns()
        interp.invoke()
        c = time.perf_counter_ns()
        logits = post(interp.get_tensor(out["index"]))
        int(logits.argmax())
        d = time.perf_counter_ns()
        return (c - b) / 1e6, (d - a) / 1e6, logits

    cold_invoke, cold_pipeline, _ = once(inputs[0])
    for i in range(1, warmup):
        once(inputs[i % len(inputs)])

    max_mhz = int(_read(FREQ.format(0, "scaling_max_freq")) or 0) / 1000.0
    before = {"stat": proc_stat(), "times": os.times(), "ru": resource.getrusage(resource.RUSAGE_SELF),
              "throttled": throttled(), "temp": temp_c()}
    sampler = Sampler(sample_interval)
    sampler.start()
    invoke_ms, pipeline_ms = [], []
    w0 = time.perf_counter_ns()
    for i in range(runs):
        a, b, _ = once(inputs[i % len(inputs)])
        invoke_ms.append(a)
        pipeline_ms.append(b)
    wall_s = (time.perf_counter_ns() - w0) / 1e9
    sampler.stop()
    after = {"stat": proc_stat(), "times": os.times(), "ru": resource.getrusage(resource.RUSAGE_SELF),
             "throttled": throttled(), "temp": temp_c()}
    mem["after_timed"] = mem_kb()

    cpu_s = (after["times"].user - before["times"].user) + (after["times"].system - before["times"].system)
    inv = stats(invoke_ms)
    report = {
        "model": os.path.basename(model), "model_bytes": os.path.getsize(model),
        "threads": threads, "resolver": resolver, "warmup": warmup, "runs": runs,
        "input": {"shape": [int(s) for s in inp["shape"]], "dtype": np.dtype(inp["dtype"]).name,
                  "quantization": [float(v) for v in inp["quantization"]]},
        "n_ops": len(interp._get_ops_details()),
        "load_ms": round(load_ms, 3),
        "cold_invoke_ms": round(cold_invoke, 4), "cold_pipeline_ms": round(cold_pipeline, 4),
        "invoke_ms": inv, "pipeline_ms": stats(pipeline_ms),
        "throughput_fps": round(1000.0 / inv["mean"], 2),
        "timed_wall_s": round(wall_s, 3),
        "cpu": {"process_util_pct": round(100.0 * cpu_s / wall_s, 1),  # 100 = one core busy
                "process_cpu_ms_per_inference": round(1000.0 * cpu_s / runs, 4),
                "user_s": round(after["times"].user - before["times"].user, 3),
                "system_s": round(after["times"].system - before["times"].system, 3),
                "system_util_pct": util_pct(before["stat"], after["stat"]),
                "voluntary_ctx_switches": after["ru"].ru_nvcsw - before["ru"].ru_nvcsw,
                "involuntary_ctx_switches": after["ru"].ru_nivcsw - before["ru"].ru_nivcsw,
                "minor_faults": after["ru"].ru_minflt - before["ru"].ru_minflt,
                "major_faults": after["ru"].ru_majflt - before["ru"].ru_majflt},
        "thermal": {"start_c": before["temp"], "end_c": after["temp"],
                    "throttled_before": before["throttled"], "throttled_after": after["throttled"],
                    **sampler.summary(max_mhz)},
    }

    if accuracy and data_x and data_y:
        x, y = np.load(data_x, mmap_mode="r"), np.load(data_y)
        a0 = time.perf_counter()
        logits = np.concatenate([once(np.array(x[i]))[2] for i in range(len(x))])
        preds = logits.argmax(1)
        report["accuracy"] = {"n": int(len(y)), "macro_f1": macro_f1(y, preds),
                              "top1_accuracy": float(np.mean(preds == y)),
                              "wall_s": round(time.perf_counter() - a0, 2)}
        if reference:
            ref = np.load(reference)
            report["parity"] = {"argmax_agreement": float(np.mean(ref.argmax(1) == preds)),
                                "n_disagree": int(np.sum(ref.argmax(1) != preds)),
                                "max_abs_logit_diff": float(np.abs(ref - logits).max()),
                                "bit_exact": bool(np.array_equal(ref, logits))}
    mem["end"] = mem_kb()
    report["memory_kb"] = mem
    report["interpreter_rss_kb"] = mem["after_load"]["VmRSS"] - mem["after_import"]["VmRSS"]
    return report


# ------------------------------------------------------------------ suite (orchestrator)
def environment(device, idle_s):
    def cmd(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    import importlib.metadata as md
    os_release = dict(line.split("=", 1) for line in (_read("/etc/os-release") or "").splitlines() if "=" in line)
    cpuinfo = _read("/proc/cpuinfo") or ""
    a, t, f = proc_stat(), [], []
    end = time.time() + idle_s
    while time.time() < end:
        t.append(temp_c())
        f.append(freq_mhz())
        time.sleep(0.5)
    return {
        "device": device,
        "model": (_read("/proc/device-tree/model") or "").replace("\x00", ""),
        "os": os_release.get("PRETTY_NAME", "").strip('"'), "kernel": platform.release(),
        "arch": platform.machine(), "cpu_count": os.cpu_count(),
        "cpu_features": next((ln.split(":", 1)[1].strip() for ln in cpuinfo.splitlines()
                              if ln.startswith("Features")), None),
        "mem_total_kb": int(next((ln.split()[1] for ln in (_read("/proc/meminfo") or "").splitlines()
                                  if ln.startswith("MemTotal")), 0)),
        "governor": _read(FREQ.format(0, "scaling_governor")),
        "freq_min_mhz": int(_read(FREQ.format(0, "scaling_min_freq")) or 0) / 1000.0,
        "freq_max_mhz": int(_read(FREQ.format(0, "scaling_max_freq")) or 0) / 1000.0,
        "python": platform.python_version(), "numpy": np.__version__,
        "ai_edge_litert": md.version("ai-edge-litert"),
        "idle_baseline": {"seconds": idle_s, "system_util_pct": util_pct(a, proc_stat()),
                          "temp_c": round(float(np.mean([v for v in t if v is not None])), 2) if t else None,
                          "freq_mhz": round(float(np.mean([v for v in f if v is not None])), 1) if f else None,
                          "loadavg": cmd("cat", "/proc/loadavg")},
        "throttled": throttled(),
        "energy": "not measured — no INA219 on the 5V rail",
    }


def build_jobs(manifest, suite, threads):
    int8 = [m for m in manifest["models"] if m["precision"] == "int8"]
    fp32 = [m for m in manifest["models"] if m["precision"] == "fp32"]
    # one INT8 model per dataset x student — latency is set by architecture, not by weights
    rep = [m for m in int8 if m["method"] == "pqk" and m["seed"] == 0]
    # methods interleaved within each seed, so any thermal/clock drift spreads across methods
    ordered = sorted(int8, key=lambda m: (m["seed"], m["dataset"], m["student"], m["method"]))
    jobs = []
    if suite in ("main", "all"):  # default deployment: XNNPACK delegate applied
        jobs += [(m, threads, "auto", True) for m in ordered]
    if suite in ("delegate", "all"):
        # builtin kernels only. XNNPACK's ARM INT8 path is not bit-identical to the PC (smoke
        # test: 8/372 flips with XNNPACK, 0 without), so accuracy is measured under both.
        jobs += [(m, threads, "no_default_delegates", True) for m in ordered]
    if suite in ("threads", "all"):
        jobs += [(m, t, "auto", False) for m in rep for t in (1, 2, 3) if t != threads]
    if suite in ("fp32", "all"):
        jobs += [(m, t, "auto", t == threads) for m in fp32 for t in (1, 2, 3, 4)]
    return jobs


def wait_cool(max_start_c, cap_s):
    t0 = time.time()
    while (temp_c() or 0) > max_start_c and time.time() - t0 < cap_s:
        time.sleep(2)
    return round(time.time() - t0, 1), temp_c()


def run_suite(args):
    bundle = os.path.expanduser(args.bundle)
    with open(os.path.join(bundle, "manifest.json")) as f:
        manifest = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    env_path = os.path.join(args.out, "environment.json")
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            json.dump(environment(args.device, args.idle_seconds), f, indent=2)
    runs_path = os.path.join(args.out, "runs.jsonl")
    done = set()
    if os.path.exists(runs_path):
        with open(runs_path) as f:
            done = {json.loads(line)["job"] for line in f if line.strip()}

    jobs = build_jobs(manifest, args.suite, args.threads)
    n_done = sum(f"{m['id']}|t{t}|{r}" in done for m, t, r, _ in jobs)
    print(f"[suite] {len(jobs)} jobs, {n_done} already done", flush=True)
    for k, (m, threads, resolver, accuracy) in enumerate(jobs, 1):
        job = f"{m['id']}|t{threads}|{resolver}"
        if job in done:
            continue
        waited, start_c = wait_cool(args.max_start_temp, args.cooldown_cap)
        ds = manifest["datasets"][m["dataset"]]
        tmp = os.path.join(args.out, ".worker.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--model", os.path.join(bundle, m["path"]),
               "--threads", str(threads), "--resolver", resolver, "--json", tmp,
               "--data-x", os.path.join(bundle, ds["x"]), "--data-y", os.path.join(bundle, ds["y"]),
               "--reference", os.path.join(bundle, m["reference"])]
        if not accuracy:
            cmd.append("--no-accuracy")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        rec = {"job": job, "id": m["id"], "precision": m["precision"], "dataset": m["dataset"],
               "student": m["student"], "method": m["method"], "seed": m["seed"], "params": m["params"],
               "size_kb": m["size_kb"], "macs": m["macs"], "pc_metrics": m["pc_metrics"],
               "cooldown_wait_s": waited, "start_temp_c": start_c}
        if proc.returncode != 0:
            rec["error"] = proc.stderr[-2000:]
            print(f"[{k}/{len(jobs)}] FAIL {job}\n{rec['error']}", flush=True)
        else:
            with open(tmp) as f:
                rec.update(json.load(f))
            os.remove(tmp)
            acc = rec.get("accuracy", {})
            par = rec.get("parity", {})
            print(f"[{k}/{len(jobs)}] {job}  p50 {rec['invoke_ms']['p50']:.2f} ms  p95 {rec['invoke_ms']['p95']:.2f}"
                  f"  cpu {rec['cpu']['process_util_pct']:.0f}%  {rec['thermal'].get('temp_c', {}).get('max')}C"
                  + (f"  F1 {acc['macro_f1']:.4f}" if acc else "")
                  + (f"  agree {par['argmax_agreement']:.4f}" if par else ""), flush=True)
        with open(runs_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print("[suite] complete", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="single .tflite to benchmark")
    p.add_argument("--bundle", help="bundle directory with manifest.json (suite mode)")
    p.add_argument("--suite", default="all", choices=["main", "threads", "fp32", "delegate", "all"])
    p.add_argument("--device", default="rpi4")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--runs", type=int, default=500)
    p.add_argument("--resolver", default="auto", choices=["auto", "no_default_delegates"])
    p.add_argument("--data-x")
    p.add_argument("--data-y")
    p.add_argument("--reference")
    p.add_argument("--no-accuracy", action="store_true")
    p.add_argument("--out", default="results")
    p.add_argument("--json", help="worker output path")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--max-start-temp", type=float, default=55.0, help="wait for SoC to cool below this")
    p.add_argument("--cooldown-cap", type=float, default=180.0)
    p.add_argument("--idle-seconds", type=float, default=10.0)
    args = p.parse_args()

    if args.bundle and not args.worker:
        return run_suite(args)
    if not args.model:
        p.error("--model or --bundle is required")
    report = bench_model(args.model, args.threads, args.warmup, args.runs, args.data_x, args.data_y,
                         args.reference, not args.no_accuracy, args.resolver)
    report["device"] = args.device
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
