"""Stage 11 — paper aggregation.

Reads results.json (from scripts/aggregate_results.py), the Raspberry Pi benchmark
(edge/results/rpi4/runs.jsonl) and the training logs, and writes LaTeX table fragments plus
numbers.json to paper/generated/. No table in the paper is typed by hand: re-run this after any
result changes.

    python scripts/aggregate.py

Conventions (stated once here, repeated in the paper):
  * mean +/- *sample* std (n-1) over 3 seeds. NOTE edge/results/rpi4/summary.md section 5 uses the
    population std (np.std) — do not copy numbers from that file into the paper.
  * `pqk` is reported at its validation-selected calibration per pair: whichever of {pqk (fixed
    lambda=1), pqk_adaptive (median-heuristic lambda)} has the higher 3-seed mean *validation*
    macro-F1, never test. The G2 table shows both so the selection is visible.
  * INT8 / on-device tables use the variant that was actually exported, which can differ from the
    selected one (flagged in the output).
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "paper", "generated")
sys.path.insert(0, os.path.join(REPO, "scripts"))

DATASETS = ["trashnet", "rps_25"]
STUDENTS = ["mobilenetv2_035", "mobilenetv3_small", "lenet5"]
EDGE_STUDENTS = ["mobilenetv2_035", "mobilenetv3_small"]
SHORT = {"mobilenetv2_035": "MobileNetV2-0.35", "mobilenetv3_small": "MobileNetV3-S", "lenet5": "LeNet-5"}
DS_SHORT = {"trashnet": "TrashNet", "rps_25": "RPS-25"}
METHODS = ["scratch", "kd", "rkd", "rbf_control", "pqk"]
# variant that was exported to INT8 and benchmarked on the Pi (scripts/dashboard.py PQK_CONFIG
# as of the export): adaptive on these two pairs, fixed lambda elsewhere.
EXPORTED_PQK = {("rps_25", "mobilenetv2_035"): "pqk_adaptive", ("trashnet", "mobilenetv3_small"): "pqk_adaptive"}


# ----------------------------------------------------------------------------- helpers
def ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def tt(m):
    return r"\texttt{%s}" % m.replace("_", r"\_")


def pm(v, nd=3):
    m, s = ms(v)
    return f"${m:.{nd}f}\\pm{s:.{nd}f}$"


def paired(a, b):
    """a, b: {seed: value}. -> (diffs list over shared seeds)"""
    return [a[s] - b[s] for s in sorted(set(a) & set(b))]


def dcell(d, nd=3):
    if not d:
        return "--"
    m, s = ms(d)
    return f"${m:+.{nd}f}\\pm{s:.{nd}f}$ ({sum(x > 0 for x in d)}/{len(d)})"


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text.rstrip() + "\n")
    print(f"wrote paper/generated/{name}")


# ----------------------------------------------------------------------------- load
with open(os.path.join(REPO, "results.json"), encoding="utf-8") as f:
    DATA = json.load(f)

RUNS = defaultdict(dict)  # (ds, student, method) -> {seed: record}
for r in DATA["students"]:
    RUNS[(r["dataset"], r["student"], r["method"])][int(r["seed"])] = r


def seeds(ds, stu, method, key="macro_f1"):
    return {s: float(r[key]) for s, r in RUNS.get((ds, stu, method), {}).items()}


# Rule #3 audit: within a (dataset, student) pair every run should share one training recipe.
# A run whose recipe differs from the pair's majority recipe is an outlier; its cells get a § marker
# in the tables and numbers.json["rule3_sensitivity"] re-computes the affected statistics without it.
_RECIPE_KEYS = ("epochs_budget", "patience", "optimizer", "lr", "weight_decay", "batch_size", "batch_norm_calibrated",
                "class_weighted_ce", "ce_label_smoothing")
RECIPE_OUTLIERS = defaultdict(set)  # (ds, student, method) -> {seed}
_by_pair = defaultdict(list)
for _r in DATA["students"]:
    if _r["student"] in STUDENTS and _r["method"] in METHODS + ["pqk_adaptive", "noent_control", "pqk_fit", "rbf_fit"]:
        _by_pair[(_r["dataset"], _r["student"])].append(_r)
for _pair, _rs in _by_pair.items():
    _sig = lambda r: tuple(r["recipe"].get(k) for k in _RECIPE_KEYS)  # noqa: E731
    _count = defaultdict(int)
    for _r in _rs:
        _count[_sig(_r)] += 1
    _major = max(_count, key=_count.get)
    for _r in _rs:
        if _sig(_r) != _major:
            RECIPE_OUTLIERS[_pair + (_r["method"],)].add(int(_r["seed"]))


# Within one (pair, method) cell the loss settings (temperature, weights) should also be identical
# across seeds. A cell whose seeds disagree is mixed; the minority seeds are listed here.
LOSS_OUTLIERS = defaultdict(set)  # (ds, student, method) -> {seed}
_cells = defaultdict(list)
for _r in DATA["students"]:
    if _r["student"] in STUDENTS and _r["method"] in METHODS + ["pqk_adaptive", "noent_control", "pqk_fit", "rbf_fit"]:
        _cells[(_r["dataset"], _r["student"], _r["method"])].append(_r)
for _key, _rs in _cells.items():
    _lc = defaultdict(int)
    for _r in _rs:
        _lc[json.dumps(_r["recipe"].get("method_loss"), sort_keys=True)] += 1
    if len(_lc) > 1:
        _maj = max(_lc, key=_lc.get)
        for _r in _rs:
            if json.dumps(_r["recipe"].get("method_loss"), sort_keys=True) != _maj:
                LOSS_OUTLIERS[_key].add(int(_r["seed"]))


def mark(ds, stu, method):
    """Footnote markers for a cell: § = a seed trained with a different budget/patience,
    ¶ = seeds trained with different loss settings (e.g. KD temperature)."""
    key = (ds, stu, method_key(ds, stu, method) if method == "pqk" else method)
    return (r"$^{\S}$" if RECIPE_OUTLIERS.get(key) else "") + (r"$^{\P}$" if LOSS_OUTLIERS.get(key) else "")


def select_pqk(ds, stu):
    """Validation-selected pqk calibration. Needs 3 seeds of both to be eligible."""
    cands = {}
    for m in ("pqk", "pqk_adaptive"):
        v = seeds(ds, stu, m, "best_val_macro_f1")
        if len(v) == 3:
            cands[m] = st.mean(v.values())
    if "pqk_adaptive" not in cands:
        return "pqk", cands
    return max(cands, key=cands.get), cands


SELECTED = {(ds, s): select_pqk(ds, s) for ds in DATASETS for s in STUDENTS}


def method_key(ds, stu, m):
    return SELECTED[(ds, stu)][0] if m == "pqk" else m


NUM = {}  # numbers.json payload

# ----------------------------------------------------------------------------- Table 1: main
def table_main():
    cols = [("trashnet", "macro_f1"), ("trashnet", "top1_accuracy"), ("rps_25", "macro_f1"), ("rps_25", "top1_accuracy")]
    L = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{TrashNet (6 classes)} & \multicolumn{2}{c}{RPS-25 (3 classes)} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Student & Method & Macro-F1 & Top-1 & Macro-F1 & Top-1 \\", r"\midrule"]
    for i, stu in enumerate(STUDENTS):
        # best mean per column within the student block
        best = {}
        for ds, key in cols:
            means = {m: st.mean(seeds(ds, stu, method_key(ds, stu, m), key).values())
                     for m in METHODS if seeds(ds, stu, method_key(ds, stu, m), key)}
            best[(ds, key)] = max(means, key=means.get)
        for j, m in enumerate(METHODS):
            cells = []
            for ds, key in cols:
                v = list(seeds(ds, stu, method_key(ds, stu, m), key).values())
                if not v:
                    cells.append("--")
                    continue
                mm, ss = ms(v)
                c = f"$\\mathbf{{{mm:.3f}}}\\pm{ss:.3f}$" if best[(ds, key)] == m else pm(v)
                if m == "pqk" and SELECTED[(ds, stu)][0] == "pqk_adaptive":
                    c += r"$^{\dagger}$"
                cells.append(c + mark(ds, stu, m))
            label = tt(m)
            first = SHORT[stu] if j == 0 else ""
            L.append(f"{first} & {label} & " + " & ".join(cells) + r" \\")
        if i < len(STUDENTS) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_main.tex", "\n".join(L))
    NUM["pqk_adaptive_selected"] = [f"{d}/{s}" for (d, s), (m, _) in SELECTED.items() if m == "pqk_adaptive"]
    NUM["pqk_selection_val"] = {f"{d}/{s}": {"chosen": m, **{k: round(v, 4) for k, v in c.items()}}
                                for (d, s), (m, c) in SELECTED.items()}


# ----------------------------------------------------------------------------- Table 2: G2
def table_g2():
    L = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"Dataset & Student & \texttt{scratch} & \texttt{rbf\_control} & $\Delta$ \texttt{pqk} ($\lambda{=}1$) & $\Delta$ \texttt{pqk} (adaptive $\lambda$) & selected \\",
         r"\midrule"]
    win_fixed = win_adapt = n_adapt = 0
    rows_num = {}
    pairs = [(ds, s) for ds in DATASETS for s in STUDENTS]
    for ds, stu in pairs:
        rbf = seeds(ds, stu, "rbf_control")
        scr = seeds(ds, stu, "scratch")
        fx = paired(seeds(ds, stu, "pqk"), rbf)
        ad_s = seeds(ds, stu, "pqk_adaptive")
        ad = paired(ad_s, rbf) if len(ad_s) == 3 else []
        win_fixed += st.mean(fx) > 0
        if ad:
            n_adapt += 1
            win_adapt += st.mean(ad) > 0
        sel = SELECTED[(ds, stu)][0]
        L.append(f"{DS_SHORT[ds]} & {SHORT[stu]} & {pm(list(scr.values()))}{mark(ds, stu, 'scratch')} & {pm(list(rbf.values()))} & "
                 f"{dcell(fx)} & {dcell(ad)} & {'adaptive' if sel == 'pqk_adaptive' else r'$\lambda{=}1$'} \\\\")
        rows_num[f"{ds}/{stu}"] = {"fixed_minus_rbf": round(st.mean(fx), 4), "fixed_wins": sum(x > 0 for x in fx),
                                   "adaptive_minus_rbf": round(st.mean(ad), 4) if ad else None,
                                   "adaptive_wins": sum(x > 0 for x in ad) if ad else None}
    L += [r"\midrule",
          f"\\multicolumn{{5}}{{@{{}}l}}{{Pairs where \\texttt{{pqk}} beats its twin on the 3-seed mean}} & "
          f"\\multicolumn{{2}}{{l}}{{{win_fixed}/{len(pairs)} ($\\lambda{{=}}1$); {win_adapt}/{n_adapt} (adaptive)}} \\\\",
          r"\bottomrule", r"\end{tabular}"]
    write("tab_g2.tex", "\n".join(L))
    NUM["g2"] = {"pairs": rows_num, "fixed_pair_wins": int(win_fixed), "n_pairs": len(pairs),
                 "adaptive_pair_wins": int(win_adapt), "n_adaptive_pairs": n_adapt}


# ----------------------------------------------------------------------------- Table 3/4: follow-up controls
def table_controls():
    L = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"Dataset & Student & \texttt{pqk} & no-CNOT & $\Delta$ val & $\Delta$ test \\", r"\midrule"]
    NUM["noent"] = {}
    for ds in DATASETS:
        for stu in EDGE_STUDENTS:
            qv, nv = seeds(ds, stu, "pqk_adaptive", "best_val_macro_f1"), seeds(ds, stu, "noent_control", "best_val_macro_f1")
            qt, nt = seeds(ds, stu, "pqk_adaptive"), seeds(ds, stu, "noent_control")
            dv, dt = paired(qv, nv), paired(qt, nt)
            L.append(f"{DS_SHORT[ds]} & {SHORT[stu]} & {pm(list(qv.values()))} & {pm(list(nv.values()))} & "
                     f"{dcell(dv)} & {dcell(dt)} \\\\")
            NUM["noent"][f"{ds}/{stu}"] = {"val_diff": round(st.mean(dv), 4), "val_wins": sum(x > 0 for x in dv),
                                           "test_diff": round(st.mean(dt), 4), "test_wins": sum(x > 0 for x in dt)}
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_noent.tex", "\n".join(L))

    L = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{\texttt{pqk}} & \multicolumn{2}{c}{\texttt{rbf\_control}} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Dataset & Student & $\Delta$ val & $\Delta$ test & $\Delta$ val & $\Delta$ test \\", r"\midrule"]
    NUM["fitted"] = {}
    better = total = 0
    for ds in DATASETS:
        for stu in EDGE_STUDENTS:
            cells, entry = [], {}
            for base, fit in (("pqk_adaptive", "pqk_fit"), ("rbf_control", "rbf_fit")):
                dv = paired(seeds(ds, stu, fit, "best_val_macro_f1"), seeds(ds, stu, base, "best_val_macro_f1"))
                dt = paired(seeds(ds, stu, fit), seeds(ds, stu, base))
                cells += [dcell(dv), dcell(dt)]
                entry[base] = {"val_diff": round(st.mean(dv), 4), "val_wins": sum(x > 0 for x in dv)}
                total += 1
                better += st.mean(dv) > 0
            L.append(f"{DS_SHORT[ds]} & {SHORT[stu]} & " + " & ".join(cells) + r" \\")
            NUM["fitted"][f"{ds}/{stu}"] = entry
    L += [r"\bottomrule", r"\end{tabular}"]
    NUM["fitted_better_arm_pairs"] = [int(better), total]
    write("tab_fitted.tex", "\n".join(L))


# ----------------------------------------------------------------------------- deploy / edge
def load_edge():
    path = os.path.join(REPO, "edge", "results", "rpi4", "runs.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        runs = [json.loads(line) for line in f if line.strip()]
    return [r for r in runs if "error" not in r]


EDGE = load_edge()
DEPLOY = defaultdict(dict)  # (ds, stu, method) -> {seed: record}
for r in DATA["deploy"]:
    DEPLOY[(r["dataset"], r["student"], r["method"])][int(r["seed"])] = r


def exported_variant(ds, stu):
    return EXPORTED_PQK.get((ds, stu), "pqk")


def numerics_scores(ds, stu, method):
    """{'pc': {seed: f1}, 'builtin': {...}, 'xnnpack': {...}, 'fp32': {...}} for the INT8 export."""
    out = {"fp32": {}, "pc": {}, "builtin": {}, "xnnpack": {}}
    for s, r in DEPLOY.get((ds, stu, method), {}).items():
        out["fp32"][s] = r["fp32"]["macro_f1"]
        out["pc"][s] = r["int8"]["macro_f1"]
    for r in EDGE:
        if (r["dataset"], r["student"], r["method"]) != (ds, stu, method):
            continue
        if r["precision"] != "int8" or r["threads"] != 4:
            continue
        if r["resolver"] == "auto":
            out["xnnpack"][r["seed"]] = r["accuracy"]["macro_f1"]
        elif r["resolver"] == "no_default_delegates":
            out["builtin"][r["seed"]] = r["accuracy"]["macro_f1"]
    return out


def table_int8():
    L = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"Dataset & Student & Exported \texttt{pqk} & PC (builtin) & Pi builtin & Pi XNNPACK \\", r"\midrule"]
    NUM["g3"] = {}
    for ds in DATASETS:
        for stu in EDGE_STUDENTS:
            pv = exported_variant(ds, stu)
            P, K = numerics_scores(ds, stu, pv), numerics_scores(ds, stu, "kd")
            cells, entry = [], {}
            for k in ("pc", "builtin", "xnnpack"):
                d = paired(P[k], K[k])
                cells.append(dcell(d))
                entry[k] = {"diff": round(st.mean(d), 4) if d else None, "wins": sum(x > 0 for x in d), "n": len(d)}
            sel = SELECTED[(ds, stu)][0]
            flag = "" if sel == pv else r"$^{\ddagger}$"
            L.append(f"{DS_SHORT[ds]} & {SHORT[stu]} & {'adaptive' if pv == 'pqk_adaptive' else r'$\lambda{=}1$'}{flag} & "
                     + " & ".join(cells) + r" \\")
            entry["exported"], entry["selected"] = pv, sel
            NUM["g3"][f"{ds}/{stu}"] = entry
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_int8.tex", "\n".join(L))

    # pooled INT8 - fp32 per method and numerics
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"Method & PC (builtin) & Pi builtin & Pi XNNPACK \\", r"\midrule"]
    NUM["int8_minus_fp32"] = {}
    for m in METHODS:
        cells, entry = [], {}
        for k in ("pc", "builtin", "xnnpack"):
            d = []
            for ds in DATASETS:
                for stu in EDGE_STUDENTS:
                    var = exported_variant(ds, stu) if m == "pqk" else m
                    sc = numerics_scores(ds, stu, var)
                    d += [sc[k][s] - sc["fp32"][s] for s in sc[k] if s in sc["fp32"]]
            cells.append(f"${st.mean(d):+.4f}$ ({sum(x > 0 for x in d)}/{len(d)})" if d else "--")
            entry[k] = {"mean": round(st.mean(d), 4), "up": sum(x > 0 for x in d), "n": len(d)} if d else None
        NUM["int8_minus_fp32"][m] = entry
        L.append(f"{tt(m)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_int8_delta.tex", "\n".join(L))

    allpc = [r["int8_minus_fp32"]["macro_f1"] for r in DATA["deploy"]]
    NUM["int8_all_models"] = {"n": len(allpc), "mean": round(st.mean(allpc), 4), "std": round(st.stdev(allpc), 4),
                              "gained": sum(x > 0 for x in allpc),
                              "sizes_kb": {f"{r['dataset']}/{r['student']}": round(r["tflite"]["size_kb"], 1)
                                           for r in DATA["deploy"]},
                              "all_integer": all(r["tflite"]["fully_integer"] for r in DATA["deploy"]),
                              "all_per_channel": all(r["tflite"]["per_channel"] for r in DATA["deploy"])}


def table_edge():
    int8_x = [r for r in EDGE if r["precision"] == "int8" and r["resolver"] == "auto" and r["threads"] == 4]
    L = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
         r"Dataset & Student & Params & INT8 KiB & MMACs & p50 ms & p95 ms & FPS & $1\!\to\!4$ thr. \\", r"\midrule"]
    NUM["edge"] = {}
    for ds in DATASETS:
        for stu in EDGE_STUDENTS:
            rr = [r for r in int8_x if (r["dataset"], r["student"]) == (ds, stu)]
            if not rr:
                continue
            p50 = st.mean(r["invoke_ms"]["p50"] for r in rr)
            p95 = st.mean(r["invoke_ms"]["p95"] for r in rr)
            fps = st.mean(r["throughput_fps"] for r in rr)
            sweep = {r["threads"]: r for r in EDGE if r["precision"] == "int8" and r["resolver"] == "auto"
                     and (r["dataset"], r["student"], r["method"], r["seed"]) == (ds, stu, "pqk", 0)}
            sp = sweep[1]["invoke_ms"]["p50"] / sweep[4]["invoke_ms"]["p50"] if 1 in sweep and 4 in sweep else float("nan")
            fp = next((r for r in EDGE if r["precision"] == "fp32" and r["threads"] == 4
                       and (r["dataset"], r["student"]) == (ds, stu)), None)
            fp_sp = fp["invoke_ms"]["p50"] / sweep[4]["invoke_ms"]["p50"] if fp and 4 in sweep else float("nan")
            L.append(f"{DS_SHORT[ds]} & {SHORT[stu]} & {rr[0]['params']:,} & {rr[0]['size_kb']:.0f} & "
                     f"{rr[0]['macs'] / 1e6:.1f} & {p50:.2f} & {p95:.2f} & {fps:.0f} & ${sp:.2f}\\times$ \\\\".replace(",", "{,}"))
            NUM["edge"][f"{ds}/{stu}"] = {"n_runs": len(rr), "p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
                                          "fps": round(fps, 1), "thread_speedup_1to4": round(sp, 3),
                                          "int8_vs_fp32_speedup_4t": round(fp_sp, 3),
                                          "macs": rr[0]["macs"], "params": rr[0]["params"],
                                          "size_kb": rr[0]["size_kb"]}
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_edge.tex", "\n".join(L))
    tmax = [r["thermal"]["temp_c"]["max"] for r in EDGE if r["thermal"].get("temp_c")]
    flagged = [r for r in EDGE if (r["thermal"].get("throttled_before") or {}).get("flags")
               or (r["thermal"].get("throttled_after") or {}).get("flags")]
    NUM["edge_overall"] = {"n_runs": len(EDGE), "max_temp_c": max(tmax), "firmware_flag_runs": len(flagged)}

    # xnnpack vs builtin agreement with PC
    NUM["xnnpack_agreement"] = {}
    for stu in EDGE_STUDENTS:
        for ds in DATASETS:
            for res, name in (("auto", "xnnpack"), ("no_default_delegates", "builtin")):
                a = [r["parity"]["argmax_agreement"] for r in EDGE if r["precision"] == "int8" and r["threads"] == 4
                     and r["resolver"] == res and (r["dataset"], r["student"]) == (ds, stu) and "parity" in r]
                NUM["xnnpack_agreement"][f"{ds}/{stu}/{name}"] = round(st.mean(a), 4) if a else None


# ----------------------------------------------------------------------------- cost
def table_cost():
    """Compute-hours accounting (rule #5). Deliberately NOT a per-method s/epoch comparison: runs
    were executed concurrently on one GPU, so logged wall-clock per epoch reflects contention, not
    method cost (a lone rbf_control epoch is ~15 s; the logged per-epoch mean is several times that)."""
    try:
        import training_cost as tc
    except Exception as e:  # logs/ may be absent on a fresh clone
        print("skipping cost table:", e)
        return
    acc = defaultdict(lambda: [0, 0.0])  # (family, kept|discarded) -> [runs, hours]
    # "kept" = the latest block of a run whose results/ entry still exists; everything else
    # (superseded re-runs, runs whose results were archived) is discarded work.
    have = {r["run_id"] for r in DATA["students"]}
    for run_id, blocks in tc.summarize(tc.parse_blocks()).items():
        fam = "pqk" if tc._is_simulator_method(run_id.split("__")[2]) else "classical"
        for i, (a, b, _epochs) in enumerate(blocks):
            kind = "kept" if (i == len(blocks) - 1 and run_id in have) else "discarded"
            acc[(fam, kind)][1] += tc._hours(a, b)
            if kind == "kept":
                acc[(fam, kind)][0] += 1
    cl, pq = acc[("classical", "kept")], acc[("pqk", "kept")]
    disc = acc[("classical", "discarded")][1] + acc[("pqk", "discarded")][1]
    L = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Run family & Runs kept & Compute-hours \\", r"\midrule",
         r"Classical (\texttt{scratch}, \texttt{kd}, \texttt{rkd}, \texttt{rbf\_control}, controls) & "
         f"{cl[0]} & {cl[1]:.1f} \\\\",
         r"\texttt{pqk} family (circuit simulator in the forward pass) & " f"{pq[0]} & {pq[1]:.1f} \\\\",
         r"\midrule", f"Discarded re-runs after pipeline fixes & -- & {disc:.1f} \\\\",
         r"\bottomrule", r"\end{tabular}"]
    write("tab_cost.tex", "\n".join(L))
    NUM["cost"] = {"classical_kept_hours": round(cl[1], 1), "pqk_family_kept_hours": round(pq[1], 1),
                   "kept_hours": round(cl[1] + pq[1], 1), "discarded_hours": round(disc, 1),
                   "total_hours": round(cl[1] + pq[1] + disc, 1), "classical_runs": cl[0], "pqk_runs": pq[0],
                   "note": "compute-hours consumed (runs overlapped in wall-clock); pqk-family hours include the CNN "
                           "and teacher, so they upper-bound simulator time (circuit ~26% of a pqk epoch in a "
                           "same-seed same-load comparison, CLAUDE.md)"}


# ----------------------------------------------------------------------------- misc
def misc():
    NUM["teachers"] = {t["dataset"]: {"macro_f1": round(t["macro_f1"], 4), "val_macro_f1": round(t["best_val_macro_f1"], 4)}
                       for t in DATA["teachers"]}
    NUM["counts"] = {"student_runs": len(DATA["students"]), "teachers": len(DATA["teachers"]),
                     "int8_exports": len(DATA["deploy"])}
    NUM["params"] = {s: next(r["params"] for r in DATA["students"] if r["student"] == s) for s in STUDENTS}
    # dataset sizes (trashnet split is cached; rps_25 caches only its 25% train subset — val/test
    # sizes are the fixed 33 / 372 images of the official folders)
    sp = {}
    for name in DATASETS:
        with open(os.path.join(REPO, "data", "splits", f"{name}.json"), encoding="utf-8") as f:
            d = json.load(f)
        cc = defaultdict(int)
        for e in d["train"]:
            cc[d["class_names"][e["label"]]] += 1
        sp[name] = {"train": len(d["train"]), "val": len(d["val"]) if "val" in d else 33,
                    "test": len(d["test"]) if "test" in d else 372, "train_per_class": dict(cc)}
    NUM["splits"] = sp
    ag = defaultdict(list)
    for r in DATA["deploy"]:
        ag[r["student"]].append(r["int8_fp32_argmax_agreement"])
    allag = [x for v in ag.values() for x in v]
    NUM["int8_fp32_agreement"] = {"all": round(st.mean(allag), 4), **{s: round(st.mean(v), 4) for s, v in ag.items()}}
    # 3-seed reference numbers quoted in the text
    ref = {}
    for ds in DATASETS:
        for stu in STUDENTS:
            for m in METHODS + ["pqk_adaptive", "noent_control", "rbf_fit", "pqk_fit"]:
                v = seeds(ds, stu, m)
                if len(v) >= 1:
                    mm, ss = ms(list(v.values()))
                    ref[f"{ds}/{stu}/{m}"] = {"macro_f1": round(mm, 4), "std": round(ss, 4), "n": len(v),
                                              "top1": round(st.mean(seeds(ds, stu, m, "top1_accuracy").values()), 4),
                                              "val": round(st.mean(seeds(ds, stu, m, "best_val_macro_f1").values()), 4)}
    NUM["cells"] = ref


def main():
    table_main()
    table_g2()
    table_controls()
    table_int8()
    table_edge()
    table_cost()
    misc()
    import aggregate_extra  # remaining tables + derived numbers (headline wording)
    aggregate_extra.build(sys.modules[__name__])
    write("numbers.json", json.dumps(NUM, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
