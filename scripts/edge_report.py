"""Summarize the on-device benchmark (edge/bench/run.py output) into comparison tables.

    python scripts/edge_report.py [--dir edge/results/rpi4]

Reads runs.jsonl + environment.json (+ camera/*/camera.json) and writes summary.md beside them.
Sections: environment, latency/CPU/memory per model family, thread scaling, fp32 vs INT8,
XNNPACK vs builtin kernels, on-device accuracy and PC parity per method, G3 re-check on device,
thermal, camera pipeline. Means are over seeds; latency is the per-run p50/p95 of 500 timed calls.
"""
import argparse
import glob
import sys
import json
import os
from collections import defaultdict

import numpy as np

# validation-selected PQK variant per pair (scripts/dashboard.py PQK_CONFIG)
PQK_REPORTED = {("rps_25", "mobilenetv2_035"): "pqk_adaptive", ("trashnet", "mobilenetv3_small"): "pqk_adaptive"}
METHODS = ["scratch", "kd", "rkd", "rbf_control", "pqk", "pqk_adaptive"]
STUDENT_SHORT = {"mobilenetv2_035": "mnv2_035", "mobilenetv3_small": "mnv3_small"}


def _ms(v):
    return f"{np.mean(v):.2f}" if len(v) else "—"


def _pm(v, nd=4):
    return f"{np.mean(v):.{nd}f} ± {np.std(v):.{nd}f}" if len(v) > 1 else (f"{v[0]:.{nd}f}" if v else "—")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="edge/results/rpi4")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    with open(os.path.join(args.dir, "environment.json")) as f:
        env = json.load(f)
    with open(os.path.join(args.dir, "runs.jsonl")) as f:
        runs = [json.loads(line) for line in f if line.strip()]
    # fp32 reference = the immutable training record (export.py verified fp32 reload equals it)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for r in runs:
        run_id = r["id"].removesuffix("__fp32")
        with open(os.path.join(root, "results", run_id, "metrics.json")) as f:
            r["fp32_macro_f1"] = json.load(f)["macro_f1"]
    errors = [r for r in runs if "error" in r]
    runs = [r for r in runs if "error" not in r]
    md = [f"# On-device benchmark — {env['model']}",
          "",
          f"{env['os']}, kernel {env['kernel']}, {env['cpu_count']} cores, governor `{env['governor']}` "
          f"({env['freq_min_mhz']:.0f}–{env['freq_max_mhz']:.0f} MHz), Python {env['python']}, "
          f"ai-edge-litert {env['ai_edge_litert']}, numpy {env['numpy']}. Idle: "
          f"{env['idle_baseline']['temp_c']} °C, {env['idle_baseline']['system_util_pct'].get('cpu')}% CPU. "
          f"Protocol: 50 warmup + 500 timed invocations, batch 1, each model in a fresh process. "
          f"**Energy: {env['energy']}.**",
          "", f"{len(runs)} benchmark runs, {len(errors)} failed."]

    int8_x = [r for r in runs if r["precision"] == "int8" and r["resolver"] == "auto" and r["threads"] == 4]
    int8_b = [r for r in runs if r["precision"] == "int8" and r["resolver"] == "no_default_delegates"]
    pairs = sorted({(r["dataset"], r["student"]) for r in int8_x})

    # ---- 1. latency / CPU / memory per pair (architecture) and per method
    md += ["", "## 1. INT8 latency, CPU and memory (XNNPACK, 4 threads)", "",
           "Latency is set by architecture: within a pair every method has the same graph, so the "
           "method spread below is measurement noise, not a method effect."]
    rows = []
    for ds, st in pairs:
        rr = [r for r in int8_x if (r["dataset"], r["student"]) == (ds, st)]
        p50 = [r["invoke_ms"]["p50"] for r in rr]
        by_m = defaultdict(list)
        for r in rr:
            by_m[r["method"]].append(r["invoke_ms"]["p50"])
        spread = max(np.mean(v) for v in by_m.values()) - min(np.mean(v) for v in by_m.values())
        rows.append([ds, STUDENT_SHORT[st], rr[0]["params"], f"{rr[0]['size_kb']:.0f}", f"{rr[0]['macs'] / 1e6:.1f}",
                     _ms(p50), _ms([r["invoke_ms"]["p95"] for r in rr]), _ms([r["invoke_ms"]["p99"] for r in rr]),
                     f"{np.mean([r['throughput_fps'] for r in rr]):.0f}",
                     _ms([r["pipeline_ms"]["p50"] for r in rr]), _ms([r["cold_invoke_ms"] for r in rr]),
                     _ms([r["load_ms"] for r in rr]),
                     f"{np.mean([r['cpu']['process_util_pct'] for r in rr]):.0f}",
                     _ms([r["cpu"]["process_cpu_ms_per_inference"] for r in rr]),
                     f"{np.mean([r['interpreter_rss_kb'] for r in rr]) / 1024:.1f}",
                     f"{spread:.3f}", len(rr)])
    md += ["", table(["dataset", "student", "params", "KB", "MMACs", "p50 ms", "p95 ms", "p99 ms", "FPS",
                      "pipeline p50", "cold ms", "load ms", "CPU %", "CPU ms/inf", "interp. RSS MB",
                      "method spread ms", "n"], rows)]

    # ---- 2. thread scaling
    md += ["", "## 2. Thread scaling (INT8, XNNPACK)", ""]
    rows = []
    for ds, st in pairs:
        rr = [r for r in runs if r["precision"] == "int8" and r["resolver"] == "auto"
              and (r["dataset"], r["student"]) == (ds, st) and r["method"] == "pqk" and r["seed"] == 0]
        base = next((r for r in rr if r["threads"] == 1), None)
        for r in sorted(rr, key=lambda r: r["threads"]):
            sp = base["invoke_ms"]["p50"] / r["invoke_ms"]["p50"] if base else float("nan")
            rows.append([ds, STUDENT_SHORT[st], r["threads"], f"{r['invoke_ms']['p50']:.2f}",
                         f"{r['invoke_ms']['p95']:.2f}", f"{sp:.2f}×", f"{r['cpu']['process_util_pct']:.0f}",
                         f"{r['cpu']['process_cpu_ms_per_inference']:.2f}",
                         r["thermal"].get("temp_c", {}).get("max")])
    md += [table(["dataset", "student", "threads", "p50 ms", "p95 ms", "speedup", "CPU %", "CPU ms/inf",
                  "max °C"], rows)]

    # ---- 3. fp32 vs int8
    md += ["", "## 3. fp32 vs INT8 (same runtime, XNNPACK)", ""]
    rows = []
    for ds, st in pairs:
        for t in (1, 4):
            f = next((r for r in runs if r["precision"] == "fp32" and (r["dataset"], r["student"]) == (ds, st)
                      and r["threads"] == t), None)
            i = next((r for r in runs if r["precision"] == "int8" and r["resolver"] == "auto" and r["method"] == "pqk"
                      and r["seed"] == 0 and (r["dataset"], r["student"]) == (ds, st) and r["threads"] == t), None)
            if not (f and i):
                continue
            rows.append([ds, STUDENT_SHORT[st], t, f"{f['size_kb']:.0f} → {i['size_kb']:.0f}",
                         f"{f['invoke_ms']['p50']:.2f} → {i['invoke_ms']['p50']:.2f}",
                         f"{f['invoke_ms']['p50'] / i['invoke_ms']['p50']:.2f}×",
                         f"{f['interpreter_rss_kb'] / 1024:.1f} → {i['interpreter_rss_kb'] / 1024:.1f}",
                         f"{f['cpu']['process_cpu_ms_per_inference']:.2f} → {i['cpu']['process_cpu_ms_per_inference']:.2f}",
                         (f"{f['accuracy']['macro_f1']:.4f} → {i['accuracy']['macro_f1']:.4f}"
                          if "accuracy" in f and "accuracy" in i else "—")])
    md += [table(["dataset", "student", "threads", "KB fp32 → int8", "p50 ms", "INT8 speedup",
                  "interp. RSS MB", "CPU ms/inf", "F1 (pqk s0)"], rows)]

    # ---- 4. XNNPACK vs builtin kernels
    md += ["", "## 4. XNNPACK vs builtin kernels (INT8, 4 threads)", ""]
    rows = []
    for ds, st in pairs:
        x = [r for r in int8_x if (r["dataset"], r["student"]) == (ds, st)]
        b = [r for r in int8_b if (r["dataset"], r["student"]) == (ds, st)]
        if not b:
            continue
        rows.append([ds, STUDENT_SHORT[st], _ms([r["invoke_ms"]["p50"] for r in x]), _ms([r["invoke_ms"]["p50"] for r in b]),
                     f"{np.mean([r['invoke_ms']['p50'] for r in b]) / np.mean([r['invoke_ms']['p50'] for r in x]):.2f}×",
                     f"{np.mean([r['parity']['argmax_agreement'] for r in x if 'parity' in r]):.4f}",
                     f"{np.mean([r['parity']['argmax_agreement'] for r in b if 'parity' in r]):.4f}",
                     f"{max(r['parity']['max_abs_logit_diff'] for r in x if 'parity' in r):.3f}",
                     f"{max(r['parity']['max_abs_logit_diff'] for r in b if 'parity' in r):.3f}"])
    md += [table(["dataset", "student", "XNNPACK p50", "builtin p50", "XNNPACK speedup", "agree w/ PC (XNNPACK)",
                  "agree w/ PC (builtin)", "max Δlogit XNNPACK", "max Δlogit builtin"], rows)]

    # ---- 5. on-device accuracy per method
    md += ["", "## 5. On-device INT8 accuracy per method (macro-F1, mean ± std over seeds)", "",
           "`device` = XNNPACK (the default deployment path); `builtin` = no default delegates; "
           "`PC` = the INT8 score in deploy.json. ★ = the validation-selected PQK variant for that pair."]
    rows = []
    for ds, st in pairs:
        for m in METHODS:
            x = sorted([r for r in int8_x if (r["dataset"], r["student"], r["method"]) == (ds, st, m)], key=lambda r: r["seed"])
            if not x:
                continue
            b = sorted([r for r in int8_b if (r["dataset"], r["student"], r["method"]) == (ds, st, m)], key=lambda r: r["seed"])
            dev = [r["accuracy"]["macro_f1"] for r in x]
            pc = [r["pc_metrics"]["macro_f1"] for r in x]
            star = " ★" if m == PQK_REPORTED.get((ds, st), "pqk") else ""
            rows.append([ds, STUDENT_SHORT[st], m + star, _pm(pc), _pm(dev),
                         _pm([r["accuracy"]["macro_f1"] for r in b]) if b else "—",
                         f"{np.mean(dev) - np.mean(pc):+.4f}",
                         f"{np.mean([r['parity']['argmax_agreement'] for r in x]):.4f}",
                         sum(r["parity"]["n_disagree"] for r in x)])
    md += [table(["dataset", "student", "method", "PC INT8", "device INT8", "device builtin", "device − PC",
                  "agreement", "flips (3 seeds)"], rows)]

    # ---- 6. G3 on device
    md += ["", "## 6. G3 re-check on device: reported `pqk` INT8 − `kd` INT8, paired by seed", ""]
    rows = []
    for ds, st in pairs:
        pm = PQK_REPORTED.get((ds, st), "pqk")
        for label, pool in (("PC", None), ("device XNNPACK", int8_x), ("device builtin", int8_b)):
            src = int8_x if pool is None else pool
            def score(m, s):
                r = next((r for r in src if (r["dataset"], r["student"], r["method"], r["seed"]) == (ds, st, m, s)), None)
                if r is None:
                    return None
                return r["pc_metrics"]["macro_f1"] if pool is None else r["accuracy"]["macro_f1"]
            d = [score(pm, s) - score("kd", s) for s in range(3) if score(pm, s) is not None and score("kd", s) is not None]
            if d:
                rows.append([ds, STUDENT_SHORT[st], pm, label, f"{np.mean(d):+.4f} ± {np.std(d):.4f}",
                             f"{sum(v > 0 for v in d)}/{len(d)}"])
    md += [table(["dataset", "student", "pqk variant", "numerics", "pqk − kd", "pqk wins"], rows)]

    # ---- 7. thermal
    tmax = [r["thermal"].get("temp_c", {}).get("max") for r in runs if r["thermal"].get("temp_c")]
    fr = [r["thermal"].get("freq_mhz", {}).get("frac_at_max") for r in runs if r["thermal"].get("freq_mhz")]
    # the sysfs read clears the firmware's sticky bits, so a flag in throttled_after happened
    # inside that run's timed window (throttled_before is read just before it)
    flagged = [(r["job"], sorted(set((r["thermal"]["throttled_before"] or {}).get("flags", [])
                                     + (r["thermal"]["throttled_after"] or {}).get("flags", []))),
                r["invoke_ms"]["p50"]) for r in runs
               if (r["thermal"].get("throttled_before") or {}).get("flags")
               or (r["thermal"].get("throttled_after") or {}).get("flags")]
    md += ["", "## 7. Thermal, clock and power", "",
           f"Max SoC temperature over all timed windows: **{max(tmax):.1f} °C** (mean of per-run maxima "
           f"{np.mean(tmax):.1f} °C), well below the Pi 4's 80 °C throttle point. Cool-down waits "
           f"(start each run below 55 °C): {sum(r['cooldown_wait_s'] for r in runs):.0f} s total. "
           f"Requested clock (cpufreq `scaling_cur_freq`) was at its maximum in {np.mean(fr):.1%} of samples "
           f"(worst run {min(fr):.1%}); the firmware's *actual* clock is not readable without root, so "
           f"firmware throttling is detected from `get_throttled` instead, which is read before and after "
           f"every timed window (the read clears its sticky bits, so a flag belongs to that window).",
           "", (f"**Firmware flags in {len(flagged)} of {len(runs)} runs:** "
                + "; ".join(f"`{j}` {f} (p50 {p:.2f} ms)" for j, f, p in flagged)
                + ". An under-voltage event means the supply sagged below ~4.63 V under load; use the "
                  "official 5.1 V / 3 A supply before measuring energy.") if flagged
           else "No firmware throttling or under-voltage flags in any run."]

    # ---- 7b. INT8 - fp32 per method under each numerics (the quantization observation)
    md += ["", "## 7b. INT8 − fp32 per method, by INT8 numerics (reported variants, 12 models each)", "",
           "fp32 = the recorded PyTorch score. PC = deploy.json (x86 LiteRT, whose default and builtin paths are "
           "bit-identical there); device builtin / XNNPACK = measured on the Pi."]
    rows = []
    for m in ("scratch", "kd", "rkd", "rbf_control", "pqk"):
        cells = [m]
        for res in (None, "no_default_delegates", "auto"):
            d = []
            for r in runs:
                if r["precision"] != "int8" or r["threads"] != 4 or r["resolver"] != (res or "auto"):
                    continue
                rm = r["method"]
                if rm.startswith("pqk") and rm != PQK_REPORTED.get((r["dataset"], r["student"]), "pqk"):
                    continue
                if ("pqk" if rm.startswith("pqk") else rm) != m:
                    continue
                q = r["pc_metrics"]["macro_f1"] if res is None else r["accuracy"]["macro_f1"]
                d.append(q - r["fp32_macro_f1"])
            cells.append(f"{np.mean(d):+.4f} (up {sum(v > 0 for v in d)}/{len(d)})" if d else "—")
        rows.append(cells)
    md += [table(["method", "PC", "device builtin", "device XNNPACK"], rows)]

    # ---- 8. camera
    cams = sorted(glob.glob(os.path.join(args.dir, "camera", "*", "camera.json")))
    if cams:
        md += ["", "## 8. Live camera pipeline (USB webcam, demonstration — frames are out of distribution)", ""]
        rows = []
        for p in cams:
            with open(p) as f:
                c = json.load(f)
            rows.append([c["id"], c.get("config", f"t{c['threads']}"), f"{c['camera']['actual'][0]}×{c['camera']['actual'][1]}@{c['camera']['actual'][2]:.0f}",
                         f"{c['capture_ms']['p50']:.1f}", f"{c['preprocess_ms']['p50']:.2f}", f"{c['invoke_ms']['p50']:.2f}",
                         f"{c['postprocess_ms']['p50']:.3f}", f"{c['processing_ms']['p50']:.2f}",
                         f"{c['achieved_fps']:.1f}", f"{c['compute_bound_fps']:.0f}", f"{c['cpu']['process_util_pct']:.0f}",
                         c["thermal"].get("temp_c", {}).get("max")])
        md += [table(["model", "threads", "camera", "capture ms", "preprocess ms", "invoke ms", "post ms", "processing ms",
                      "achieved FPS", "compute-bound FPS", "CPU %", "max °C"], rows)]

    if errors:
        md += ["", "## Failed runs", ""] + [f"- `{r['job']}`" for r in errors]
    out = os.path.join(args.dir, "summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
