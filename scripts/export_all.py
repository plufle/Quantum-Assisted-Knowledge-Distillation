"""Export every *reported* checkpoint to ONNX + fully-integer INT8 TFLite (Stage 8).

Reported = what the dashboard shows: mobilenetv2_035 / mobilenetv3_small on both datasets,
the five final methods, plus the pqk_adaptive runs the dashboard reports as `pqk` on the
pairs where validation selected adaptive lambda. lenet5, the width probes, and the lambda/
tau/gamma/ZZ screens are deliberately skipped — they are not in the results table.

Sequential, single process (TensorFlow is imported once), skips runs already exported, and
records failures without stopping so one bad checkpoint cannot strand the rest.

    python scripts/export_all.py
"""
import json
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from qakd.deploy.export import EXPORT_ROOT, export_run  # noqa: E402

STUDENTS = {"mobilenetv2_035", "mobilenetv3_small"}
FINAL_METHODS = {"scratch", "kd", "rkd", "rbf_control", "pqk"}
ADAPTIVE_REPORTED = {("rps_25", "mobilenetv2_035"), ("trashnet", "mobilenetv3_small")}


def reported_run_ids():
    with open(os.path.join(REPO_ROOT, "results.json")) as f:
        students = json.load(f)["students"]
    ids = []
    for r in students:
        if r["student"] not in STUDENTS:
            continue
        if r["method"] in FINAL_METHODS or (
            r["method"] == "pqk_adaptive" and (r["dataset"], r["student"]) in ADAPTIVE_REPORTED
        ):
            ids.append(r["run_id"])
    return sorted(ids)


def main():
    import tensorflow as tf

    ids = reported_run_ids()
    done = [i for i in ids if os.path.exists(os.path.join(EXPORT_ROOT, i, "deploy.json"))]
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} reported runs: {len(done)} already exported, {len(todo)} to do", flush=True)
    failures = []
    for n, run_id in enumerate(todo, 1):
        try:
            r = export_run(run_id)
            print(f"[{n}/{len(todo)}] OK   {run_id}  fp32 {r['fp32']['macro_f1']:.4f} -> "
                  f"int8 {r['int8']['macro_f1']:.4f} ({r['int8_minus_fp32']['macro_f1']:+.4f})  "
                  f"{r['tflite']['size_kb']:.0f} KB", flush=True)
        except Exception as e:  # keep going; report at the end
            failures.append((run_id, repr(e)))
            print(f"[{n}/{len(todo)}] FAIL {run_id}: {e!r}", flush=True)
            traceback.print_exc()
        finally:
            tf.keras.backend.clear_session()
    print(f"=== EXPORT COMPLETE: {len(todo) - len(failures)} ok, {len(failures)} failed ===", flush=True)
    for run_id, err in failures:
        print(f"  FAILED {run_id}: {err}", flush=True)


if __name__ == "__main__":
    main()
