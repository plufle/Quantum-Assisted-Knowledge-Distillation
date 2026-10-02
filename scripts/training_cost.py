"""Training-cost accounting (CLAUDE.md rule #5: log GPU-hours + simulator-hours, report it).

Recovers per-run wall-clock from the epoch timestamps already in logs/*.log, so it covers
every run ever executed without needing the trainer to have recorded durations.

A run_id can appear more than once in the logs (re-runs after a pipeline fix), so blocks
are split on the epoch counter resetting to 1. That lets us separate two different numbers
that rule #5 cares about:
  * cost of the results we actually kept  (latest block per run_id)
  * total machine cost including discarded work (every block)

Runs executed in parallel overlap in wall-clock, so summed hours are compute-hours
consumed, not elapsed time on the clock.

Usage: python scripts/training_cost.py
"""
import glob
import os
import re
from collections import defaultdict
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LINE_RE = re.compile(
    r"\[ (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \].*?\[(?P<run_id>[\w.]+__[\w.]+__[\w.]+__s\d+)\] epoch (?P<epoch>\d+)/"
)
# Every pqk variant (pqk, pqk_adaptive, pqk_zz, and the pqk_t3/g05/g20/d2 screens) runs the
# circuit simulator in its forward pass — matched by prefix so new variants are not silently
# counted as classical. NOTE: this splits runs by *type*, not by resource. A PQK run's hours
# include the CNN forward/backward and the frozen teacher, so "PQK-run hours" is an upper
# bound on simulator time, not a measurement of it. Separating the two needs the kernel call
# timed inside trainer.py. As a rough guide, same-seed same-load comparison put the circuit at
# ~26% of a pqk epoch (19.4 vs 15.4 s/epoch against rbf_control).
def _is_simulator_method(method):
    return method.startswith("pqk")


def parse_blocks():
    """-> list of (run_id, start, end, epochs), one entry per contiguous training block."""
    blocks = []
    open_block = {}
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "logs", "*.log"))):
        with open(path, errors="replace") as f:
            for line in f:
                m = LINE_RE.search(line)
                if not m:
                    continue
                run_id, epoch = m.group("run_id"), int(m.group("epoch"))
                ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
                if epoch == 1 and run_id in open_block:
                    blocks.append(open_block.pop(run_id))
                if run_id not in open_block:
                    open_block[run_id] = [run_id, ts, ts, 0]
                open_block[run_id][2] = ts
                open_block[run_id][3] = epoch
    blocks.extend(open_block.values())
    return [tuple(b) for b in blocks]


def summarize(blocks):
    by_run = defaultdict(list)
    for run_id, start, end, epochs in blocks:
        by_run[run_id].append((start, end, epochs))
    for v in by_run.values():
        v.sort()
    return by_run


def _hours(start, end):
    return (end - start).total_seconds() / 3600.0


def main():
    blocks = parse_blocks()
    if not blocks:
        print("No epoch lines found in logs/ — nothing to account for.")
        return
    by_run = summarize(blocks)

    kept_sim = kept_gpu = total_sim = total_gpu = 0.0
    rows = []
    for run_id, runs in sorted(by_run.items()):
        method = run_id.split("__")[2]
        is_sim = _is_simulator_method(method)
        for i, (start, end, epochs) in enumerate(runs):
            h = _hours(start, end)
            if is_sim:
                total_sim += h
            else:
                total_gpu += h
            if i == len(runs) - 1:  # latest block = the result we kept
                if is_sim:
                    kept_sim += h
                else:
                    kept_gpu += h
                rows.append((run_id, epochs, h, h * 3600 / max(epochs, 1), len(runs)))

    print(f"{'run_id':<48} {'epochs':>6} {'hours':>7} {'s/epoch':>8} {'attempts':>8}")
    for run_id, epochs, h, sec_ep, attempts in rows:
        print(f"{run_id:<48} {epochs:>6} {h:>7.2f} {sec_ep:>8.1f} {attempts:>8}")

    discarded = (total_gpu + total_sim) - (kept_gpu + kept_sim)
    print()
    print("Cost of kept results   — classical-run hours %.2f | PQK-run hours %.2f | total %.2f"
          % (kept_gpu, kept_sim, kept_gpu + kept_sim))
    print("Total incl. discarded  — classical-run hours %.2f | PQK-run hours %.2f | total %.2f"
          % (total_gpu, total_sim, total_gpu + total_sim))
    print("PQK-run hours include CNN + teacher compute; est. simulator share ~26%% -> ~%.2f h kept"
          % (kept_sim * 0.26))
    print("Discarded (re-runs after pipeline fixes): %.2f hours" % discarded)
    print()
    print("Note: runs executed in parallel, so these are compute-hours consumed, not elapsed "
          "wall-clock. PQK-run hours = all pqk* variants (lightning.qubit in the forward pass).")


if __name__ == "__main__":
    main()
