"""Consolidates checkpoints/teachers/*.json and results/*/metrics.json into a single
results.json at the repo root — the dashboard reads this one file instead of scanning
many small ones (fewer files to track in git; one read instead of N).

Run this after any training run(s) to refresh it:
    python scripts/aggregate_results.py
"""
import glob
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(REPO_ROOT, "results.json")


def build():
    teachers = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "checkpoints", "teachers", "*.json"))):
        with open(path) as f:
            teachers.append(json.load(f))

    students = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "results", "*", "metrics.json"))):
        with open(path) as f:
            students.append(json.load(f))

    # Stage 8: fp32/INT8 reports from qakd.deploy.export. exports/ itself is gitignored
    # (binary ONNX/TFLite), so the small JSON summaries are carried here for the dashboard.
    deploy = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "exports", "*", "deploy.json"))):
        with open(path) as f:
            deploy.append(json.load(f))

    return {"teachers": teachers, "students": students, "deploy": deploy}


if __name__ == "__main__":
    data = build()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {len(data['teachers'])} teacher(s), {len(data['students'])} student run(s) and "
          f"{len(data['deploy'])} INT8 export(s) to {OUTPUT_PATH}")
