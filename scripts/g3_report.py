"""G3 — fp32 vs INT8 for every reported model, and the G3 gate: `pqk` INT8 vs `kd` INT8 at
matched KB. Reads exports/*/deploy.json (written by qakd.deploy.export); never re-runs anything.

Uses the same per-pair pqk calibration the dashboard reports (validation-selected): adaptive
lambda on rps_25/mobilenetv2_035 and trashnet/mobilenetv3_small, fixed lambda elsewhere.

    python scripts/g3_report.py
"""
import glob
import json
import os
import statistics as st
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PQK_CONFIG = {("rps_25", "mobilenetv2_035"): "pqk_adaptive",
              ("trashnet", "mobilenetv3_small"): "pqk_adaptive"}
METHODS = ["scratch", "kd", "rkd", "rbf_control", "pqk"]


def _ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def load():
    cells = defaultdict(list)
    for path in glob.glob(os.path.join(REPO_ROOT, "exports", "*", "deploy.json")):
        with open(path) as f:
            r = json.load(f)
        key = (r["dataset"], r["student"])
        method = r["method"]
        if method == "pqk_adaptive":
            if PQK_CONFIG.get(key) != "pqk_adaptive":
                continue
            method = "pqk"
        elif method == "pqk" and PQK_CONFIG.get(key) == "pqk_adaptive":
            continue
        cells[key + (method,)].append(r)
    return cells


def main():
    cells = load()
    pairs = sorted({(d, s) for d, s, _ in cells})
    print("fp32 vs INT8 macro-F1 (mean +/- std over seeds)")
    print(f"{'pair':<30}{'method':<13}{'n':>2} {'fp32':>15} {'int8':>15} {'int8-fp32':>16} {'KB':>7}")
    for d, s in pairs:
        for m in METHODS:
            rs = cells.get((d, s, m), [])
            if not rs:
                continue
            f = _ms([r["fp32"]["macro_f1"] for r in rs])
            q = _ms([r["int8"]["macro_f1"] for r in rs])
            dlt = _ms([r["int8_minus_fp32"]["macro_f1"] for r in rs])
            kb = st.mean(r["tflite"]["size_kb"] for r in rs)
            print(f"{d + '/' + s:<30}{m:<13}{len(rs):>2} {f[0]:.4f}+/-{f[1]:.4f} {q[0]:.4f}+/-{q[1]:.4f} "
                  f"{dlt[0]:+.4f}+/-{dlt[1]:.4f} {kb:>7.1f}")
        print()

    print("G3: pqk INT8 vs kd INT8 at matched KB")
    for d, s in pairs:
        p, k = cells.get((d, s, "pqk"), []), cells.get((d, s, "kd"), [])
        if len(p) < 3 or len(k) < 3:
            print(f"  {d}/{s}: incomplete (pqk n={len(p)}, kd n={len(k)})")
            continue
        p_kb, k_kb = st.mean(r["tflite"]["size_kb"] for r in p), st.mean(r["tflite"]["size_kb"] for r in k)
        p8, k8 = _ms([r["int8"]["macro_f1"] for r in p]), _ms([r["int8"]["macro_f1"] for r in k])
        pd_, kd_ = st.mean(r["int8_minus_fp32"]["macro_f1"] for r in p), st.mean(r["int8_minus_fp32"]["macro_f1"] for r in k)
        verdict = "PASS" if p8[0] > k8[0] else "FAIL"
        print(f"  {d}/{s}: pqk int8 {p8[0]:.4f}+/-{p8[1]:.4f} vs kd int8 {k8[0]:.4f}+/-{k8[1]:.4f} "
              f"({p8[0]-k8[0]:+.4f}) at {p_kb:.0f} vs {k_kb:.0f} KB -> {verdict} | "
              f"quantization loss: pqk {pd_:+.4f}, kd {kd_:+.4f}")


if __name__ == "__main__":
    main()
