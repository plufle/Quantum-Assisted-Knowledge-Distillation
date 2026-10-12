"""Companion to scripts/aggregate.py: the remaining paper tables and the derived numbers behind the
headline wording (pqk vs kd, seed variance). Not run directly — aggregate.main() calls build(agg).

Every figure here is computed from results.json / the Pi benchmark / the configs; none is typed.
"""
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

SEEDS3 = (0, 1, 2)


def build(agg):
    A = agg
    NUM, write, tt, pm, ms, paired, dcell = A.NUM, A.write, A.tt, A.pm, A.ms, A.paired, A.dcell
    PAIRS = [(ds, s) for ds in A.DATASETS for s in A.STUDENTS]
    EDGE_PAIRS = [(ds, s) for ds in A.DATASETS for s in A.EDGE_STUDENTS]
    METH5 = ["scratch", "kd", "rkd", "rbf_control", "pqk"]
    NUMERICS = [("pc", "PC (builtin)"), ("builtin", "Pi builtin"), ("xnnpack", "Pi XNNPACK")]

    def pair_name(ds, s):
        return f"{A.DS_SHORT[ds]} & {A.SHORT[s]}"

    def sd(v):
        return st.stdev(v) if len(v) > 1 else float("nan")

    def sci(x):
        m, e = f"{x:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"

    def fp32(ds, s, m, key="macro_f1"):
        return A.seeds(ds, s, A.method_key(ds, s, m), key)

    def int8(ds, s, m, num):
        var = A.exported_variant(ds, s) if m == "pqk" else m
        return A.numerics_scores(ds, s, var)[num]

    # ------------------------------------------------------------------ pqk vs kd (headline)
    L = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r" & & fp32 & \multicolumn{3}{c}{INT8} \\", r"\cmidrule(lr){4-6}",
         r"Dataset & Student & test macro-F1 & PC (builtin) & Pi builtin & Pi XNNPACK \\", r"\midrule"]
    out = {"fp32": {}, "int8": {k: {} for k, _ in NUMERICS}}
    ahead = {"fp32": 0, **{k: 0 for k, _ in NUMERICS}}
    n_int8 = 0
    for ds, s in PAIRS:
        d32 = paired(fp32(ds, s, "pqk"), fp32(ds, s, "kd"))
        ahead["fp32"] += st.mean(d32) > 0
        out["fp32"][f"{ds}/{s}"] = {"diff": round(st.mean(d32), 4), "wins": sum(x > 0 for x in d32),
                                    "pqk_variant": A.method_key(ds, s, "pqk")}
        cells = [dcell(d32) + A.mark(ds, s, "kd")]
        if s in A.EDGE_STUDENTS:
            n_int8 += 1
            for k, _ in NUMERICS:
                d = paired(int8(ds, s, "pqk", k), int8(ds, s, "kd", k))
                ahead[k] += st.mean(d) > 0
                out["int8"][k][f"{ds}/{s}"] = {"diff": round(st.mean(d), 4), "wins": sum(x > 0 for x in d)}
                cells.append(dcell(d))
        else:
            cells += ["--"] * 3
        L.append(f"{pair_name(ds, s)} & " + " & ".join(cells) + r" \\")
    L += [r"\midrule",
          r"\multicolumn{2}{@{}l}{Pairs where \texttt{pqk} is ahead on the mean} & "
          f"{ahead['fp32']}/{len(PAIRS)} & " + " & ".join(f"{ahead[k]}/{n_int8}" for k, _ in NUMERICS) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    write("tab_pqk_kd.tex", "\n".join(L))
    out["ahead"] = {"fp32": [int(ahead["fp32"]), len(PAIRS)], **{k: [int(ahead[k]), n_int8] for k, _ in NUMERICS}}
    NUM["pqk_vs_kd"] = out

    # ------------------------------------------------------------------ seed variance
    def std_row(getter, ds, s, extra_variants):
        row = {m: sd(list(getter(m).values())) if getter(m) else float("nan") for m in METH5}
        for name, key in extra_variants:
            v = A.seeds(ds, s, key)
            row[name] = sd(list(v.values())) if len(v) == 3 else float("nan")
        return row

    def fmt_std(row, cols):
        vals = {c: row[c] for c in cols if not math.isnan(row[c])}
        lo = min(vals.values())
        return [("--" if math.isnan(row[c]) else (f"$\\mathbf{{{row[c]:.3f}}}$" if row[c] == lo else f"${row[c]:.3f}$"))
                for c in cols]

    def pooled(rows, col):
        v = [r[col] for r in rows if not math.isnan(r[col])]
        return math.sqrt(st.mean(x * x for x in v)) if v else float("nan")

    def lower(rows, a, b):
        ok = [(r[a] < r[b]) for r in rows if not (math.isnan(r[a]) or math.isnan(r[b]))]
        return int(sum(ok)), len(ok)

    # fp32: both pqk calibrations shown, the validation-selected one is what the headline uses
    cols = ["scratch", "kd", "rkd", "rbf_control", "pqk_fixed", "pqk_adaptive", "pqk"]
    head = ["\\texttt{scratch}", "\\texttt{kd}", "\\texttt{rkd}", "\\texttt{rbf\\_control}",
            r"\texttt{pqk} $\lambda{=}1$", r"\texttt{pqk} adaptive", r"\texttt{pqk} selected"]
    L = [r"\begin{tabular}{@{}ll" + "c" * len(cols) + r"@{}}", r"\toprule",
         "Dataset & Student & " + " & ".join(head) + r" \\", r"\midrule"]
    rows32, var = [], {"fp32": {}}
    for ds, s in PAIRS:
        r = std_row(lambda m: fp32(ds, s, m), ds, s, [("pqk_fixed", "pqk"), ("pqk_adaptive", "pqk_adaptive")])
        r["pqk"] = sd(list(fp32(ds, s, "pqk").values()))
        rows32.append(r)
        var["fp32"][f"{ds}/{s}"] = {k: (None if math.isnan(v) else round(v, 4)) for k, v in r.items()}
        # bold only the minimum among the five methods using the selected pqk
        base = {m: r[m] for m in METH5 if not math.isnan(r[m])}
        lo = min(base.values())
        cells = []
        for c in cols:
            v = r[c]
            cells.append("--" if math.isnan(v) else (f"$\\mathbf{{{v:.3f}}}$" if (c in base and v == lo) else f"${v:.3f}$")
                         + (A.mark(ds, s, c) if c in ("scratch", "kd", "rkd", "rbf_control") else ""))
        L.append(f"{pair_name(ds, s)} & " + " & ".join(cells) + r" \\")
    cnt = {k: lower(rows32, "pqk", k) for k in ("scratch", "rbf_control", "kd", "rkd")}
    both = sum(1 for r in rows32 if r["pqk"] < r["scratch"] and r["pqk"] < r["rbf_control"])
    L.append(r"\midrule")
    L.append(r"Pooled std (root-mean-square over pairs) & & " + " & ".join(
        f"${pooled(rows32, c):.3f}$" for c in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_var_fp32.tex", "\n".join(L))
    var["fp32_summary"] = {"selected_lower_than": {k: list(v) for k, v in cnt.items()}, "lower_than_both_scratch_rbf": both,
                           "n_pairs": len(rows32),
                           "pooled": {c: round(pooled(rows32, c), 4) for c in cols},
                           "fixed_lambda_lower_than_scratch": list(lower(rows32, "pqk_fixed", "scratch")),
                           "fixed_lambda_lower_than_rbf": list(lower(rows32, "pqk_fixed", "rbf_control")),
                           "sign_test_p_4of6": round(2 * sum(math.comb(6, i) for i in range(4, 7)) / 2 ** 6, 3)}

    # INT8: exported pqk variant per pair (flagged where it differs from the selected one)
    L = [r"\begin{tabular}{@{}lllccccc@{}}", r"\toprule",
         r"INT8 numerics & Dataset & Student & \texttt{scratch} & \texttt{kd} & \texttt{rkd} & \texttt{rbf\_control} & \texttt{pqk} \\",
         r"\midrule"]
    var["int8"], var["int8_summary"] = {}, {}
    for bi, (k, label) in enumerate(NUMERICS):
        rows = []
        for j, (ds, s) in enumerate(EDGE_PAIRS):
            r = {m: sd(list(int8(ds, s, m, k).values())) for m in METH5}
            rows.append(r)
            var["int8"].setdefault(k, {})[f"{ds}/{s}"] = {m: round(v, 4) for m, v in r.items()}
            lo = min(r.values())
            cells = [f"$\\mathbf{{{r[m]:.3f}}}$" if r[m] == lo else f"${r[m]:.3f}$" for m in METH5]
            flag = r"$^{\ddagger}$" if A.exported_variant(ds, s) != A.SELECTED[(ds, s)][0] else ""
            L.append(f"{label if j == 0 else ''} & {pair_name(ds, s)} & " + " & ".join(cells[:-1]) + f" & {cells[-1]}{flag} \\\\")
        c = {kk: lower(rows, "pqk", kk) for kk in ("scratch", "rbf_control", "kd", "rkd")}
        var["int8_summary"][k] = {"lower_than": {kk: list(v) for kk, v in c.items()},
                                  "lower_than_both_scratch_rbf": sum(1 for r in rows if r["pqk"] < r["scratch"] and r["pqk"] < r["rbf_control"]),
                                  "pooled": {m: round(pooled(rows, m), 4) for m in METH5},
                                  "trashnet_pairs_lower_than_both": sum(
                                      1 for (ds, s), r in zip(EDGE_PAIRS, rows)
                                      if ds == "trashnet" and r["pqk"] < r["scratch"] and r["pqk"] < r["rbf_control"])}
        if bi < len(NUMERICS) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_var_int8.tex", "\n".join(L))
    NUM["variance"] = var

    # ------------------------------------------------------------------ validation selection of lambda
    L = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"Dataset & Student & $\lambda{=}1$ (val.) & adaptive (val.) & selected \\", r"\midrule"]
    for ds, s in PAIRS:
        fx = list(A.seeds(ds, s, "pqk", "best_val_macro_f1").values())
        ad = A.seeds(ds, s, "pqk_adaptive", "best_val_macro_f1")
        adc = pm(list(ad.values())) if len(ad) == 3 else f"-- ($n{{=}}{len(ad)}$)"
        sel = A.SELECTED[(ds, s)][0]
        L.append(f"{pair_name(ds, s)} & {pm(fx)} & {adc} & {'adaptive' if sel == 'pqk_adaptive' else r'$\lambda{=}1$'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_val_select.tex", "\n".join(L))

    # ------------------------------------------------------------------ INT8 export engineering
    groups = defaultdict(list)
    for r in A.DATA["deploy"]:
        groups[(r["dataset"], r["student"])].append(r)
    L = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
         r"Dataset & Student & Models & INT8 KiB & Conv ops & Integer-only & Per-channel & ONNX $\max|\Delta$logit$|$ & fp32 reload exact & INT8$-$fp32 \\",
         r"\midrule"]
    exp = {}
    for ds, s in EDGE_PAIRS:
        g = groups[(ds, s)]
        ops = {json.dumps(r["tflite"]["ops"], sort_keys=True) for r in g}
        d = [r["int8_minus_fp32"]["macro_f1"] for r in g]
        exact = sum(abs(r["recorded"]["macro_f1"] - r["fp32"]["macro_f1"]) < 1e-9 for r in g)
        L.append(f"{pair_name(ds, s)} & {len(g)} & {st.mean(r['tflite']['size_kb'] for r in g):.0f} & {g[0]['tflite']['n_conv']} & "
                 f"{sum(r['tflite']['fully_integer'] for r in g)}/{len(g)} & {sum(r['tflite']['per_channel'] for r in g)}/{len(g)} & "
                 f"{sci(max(r['onnx']['max_abs_logit_diff'] for r in g))} & {exact}/{len(g)} & ${st.mean(d):+.4f}\\pm{sd(d):.4f}$ \\\\")
        exp[f"{ds}/{s}"] = {"models": len(g), "kb": round(st.mean(r["tflite"]["size_kb"] for r in g), 1),
                            "n_conv": g[0]["tflite"]["n_conv"], "identical_op_set_across_models": len(ops) == 1,
                            "tflite_ops": g[0]["tflite"]["ops"], "onnx_opset": g[0]["onnx"]["opset"],
                            "onnx_ops": g[0]["onnx"]["ops"],
                            "onnx_argmax_agreement_min": min(r["onnx"]["argmax_agreement"] for r in g),
                            "n_calibration": g[0]["tflite"]["n_calibration"],
                            "versions": g[0]["versions"]}
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_export.tex", "\n".join(L))
    NUM["export"] = exp

    # ------------------------------------------------------------------ pqk variant screens (TrashNet / MobileNetV3-S)
    ds, s = "trashnet", "mobilenetv3_small"
    variants = [("pqk", r"fixed $\lambda{=}1$ (pre-specified)"), ("pqk_adaptive", r"adaptive $\lambda$ (median heuristic)"),
                ("pqk_t3", r"adaptive, $\tau{=}3$"), ("pqk_g05", r"adaptive, $\gamma{=}0.5$"),
                ("pqk_g20", r"adaptive, $\gamma{=}2$"), ("pqk_d2", r"adaptive, depth 2"),
                ("pqk_zz", r"adaptive + $\langle Z_iZ_j\rangle$ correlators"), ("noent_control", "no-CNOT circuit"),
                ("rbf_control", "RBF on the 8 angles"), ("kd", "reference"), ("scratch", "reference")]
    L = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"Variant & Change & seeds & seed-0 test F1 / val F1 & 3-seed test F1 \\", r"\midrule"]
    scr = {}
    for m, desc in variants:
        t, v = A.seeds(ds, s, m), A.seeds(ds, s, m, "best_val_macro_f1")
        if 0 not in t:
            continue
        three = pm(list(t.values())) if len(t) == 3 else "--"
        L.append(f"{tt(m)} & {desc} & {len(t)} & ${t[0]:.4f}$ / ${v[0]:.4f}$ & {three} \\\\")
        scr[m] = {"n": len(t), "seed0_test": round(t[0], 4), "seed0_val": round(v[0], 4),
                  "mean_test": round(st.mean(t.values()), 4)}
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_screens.tex", "\n".join(L))
    NUM["screens_trashnet_mnv3"] = scr

    # ------------------------------------------------------------------ on-device accuracy per method
    L = [r"\begin{tabular}{@{}lllccccc@{}}", r"\toprule",
         r"Dataset & Student & Method & fp32 & PC INT8 & Pi builtin & Pi XNNPACK & XNNPACK agree. \\", r"\midrule"]
    dev = {}
    for ds, s in EDGE_PAIRS:
        for j, m in enumerate(METH5):
            var_ = A.exported_variant(ds, s) if m == "pqk" else m
            sc = A.numerics_scores(ds, s, var_)
            if not all(sc[k] for k in ("fp32", "pc", "builtin", "xnnpack")):
                continue
            agree = [r["parity"]["argmax_agreement"] for r in A.EDGE if r["precision"] == "int8" and r["threads"] == 4
                     and r["resolver"] == "auto" and (r["dataset"], r["student"], r["method"]) == (ds, s, var_)]
            lab = tt(m) + (r"$^{\dagger}$" if (m == "pqk" and var_ == "pqk_adaptive") else "")
            L.append(f"{pair_name(ds, s) if j == 0 else ' & '} & {lab} & " +
                     " & ".join(pm(list(sc[k].values())) for k in ("fp32", "pc", "builtin", "xnnpack")) +
                     f" & ${st.mean(agree):.3f}$ \\\\")
            dev[f"{ds}/{s}/{m}"] = {k: round(st.mean(sc[k].values()), 4) for k in ("fp32", "pc", "builtin", "xnnpack")}
        if (ds, s) != EDGE_PAIRS[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_device_acc.tex", "\n".join(L))
    NUM["device_acc"] = dev

    # ------------------------------------------------------------------ camera pipeline
    rows = []
    for p in sorted(glob.glob(os.path.join(A.REPO, "edge", "results", "rpi4", "camera", "*", "camera.json"))):
        with open(p, encoding="utf-8") as f:
            c = json.load(f)
        rows.append(c)
    L = [r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
         r"Student & Method & TFLite thr. & CV thr. & preproc. & invoke & post & processing & FPS & CPU \% & max $^\circ$C \\", r"\midrule"]
    cam = []
    for c in sorted(rows, key=lambda c: (c["student"], c["method"], c["threads"], str(c["cv_threads"]))):
        L.append(f"{A.SHORT[c['student']]} & {tt(c['method'])} & {c['threads']} & {c['cv_threads']} & "
                 f"{c['preprocess_ms']['p50']:.1f} & {c['invoke_ms']['p50']:.1f} & {c['postprocess_ms']['p50']:.2f} & "
                 f"{c['processing_ms']['p50']:.1f} & {c['achieved_fps']:.1f} & {c['cpu']['process_util_pct']:.0f} & "
                 f"{c['thermal'].get('temp_c', {}).get('max', float('nan')):.1f} \\\\")
        cam.append({"id": c["id"], "config": c["config"], "threads": c["threads"], "cv_threads": c["cv_threads"],
                    "preprocess_p50": c["preprocess_ms"]["p50"], "invoke_p50": c["invoke_ms"]["p50"],
                    "post_p50": c["postprocess_ms"]["p50"], "processing_p50": c["processing_ms"]["p50"],
                    "capture_p50": c["capture_ms"]["p50"], "fps": c["achieved_fps"],
                    "compute_bound_fps": c["compute_bound_fps"], "cpu_pct": c["cpu"]["process_util_pct"]})
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_camera.tex", "\n".join(L))
    NUM["camera"] = cam

    # ------------------------------------------------------------------ datasets
    sp = NUM["splits"]
    tn, rp = sp["trashnet"], sp["rps_25"]
    lo, hi = min(tn["train_per_class"].items(), key=lambda kv: kv[1]), max(tn["train_per_class"].items(), key=lambda kv: kv[1])
    L = [r"\begin{tabular}{@{}lrrrrrll@{}}", r"\toprule",
         r"Dataset & Classes & Train & Val & Test & Smallest / largest train class & Train sampler & Primary metric \\", r"\midrule",
         f"TrashNet & 6 & {tn['train']:,} & {tn['val']} & {tn['test']} & {lo[0]} {lo[1]} / {hi[0]} {hi[1]} & class-balanced (weighted) & macro-F1 \\\\",
         f"RPS-25 & 3 & {rp['train']} & {rp['val']} & {rp['test']} & 210 / 210 (balanced) & shuffle & top-1 accuracy \\\\",
         r"\bottomrule", r"\end{tabular}"]
    write("tab_data.tex", "\n".join(L).replace(",", "{,}"))

    # ------------------------------------------------------------------ hyper-parameters actually used (from run recipes)
    try:
        import yaml
    except ImportError:
        yaml = None
    by = defaultdict(lambda: defaultdict(set))  # (ds, student) -> field -> values
    for r in A.DATA["students"]:
        if r["student"] not in A.STUDENTS or r["method"] not in METH5 + ["pqk_adaptive", "noent_control", "pqk_fit", "rbf_fit"]:
            continue
        rc = r["recipe"]
        for f_ in ("epochs_budget", "patience", "lr", "weight_decay", "batch_size", "batch_norm_calibrated"):
            by[(r["dataset"], r["student"])][f_].add(rc.get(f_))
    audit = {f"{ds}/{s}": {k: sorted(map(str, v)) for k, v in by[(ds, s)].items() if len(v) > 1} for ds, s in PAIRS}
    NUM["rule3_recipe_mismatch"] = {k: v for k, v in audit.items() if v}
    dcfg = {}
    if yaml:
        for ds in A.DATASETS:
            with open(os.path.join(A.REPO, "configs", "dataset", f"{ds}.yaml"), encoding="utf-8") as f:
                dcfg[ds] = yaml.safe_load(f)
    L = [r"\begin{tabular}{@{}llrrrcccc@{}}", r"\toprule",
         r"Dataset & Student & Epochs & Patience & LR & BN stats & RRC min scale & RandAugment (ops, mag.) \\", r"\midrule"]
    for ds, s in PAIRS:
        f_ = by[(ds, s)]
        ep = "/".join(str(x) for x in sorted(f_["epochs_budget"]))
        pa = "/".join(str(x) for x in sorted(f_["patience"]))
        lr = sorted(f_["lr"])[0]
        bn = "per-batch + recal." if True in f_["batch_norm_calibrated"] else "running"
        d = dcfg.get(ds, {})
        L.append(f"{pair_name(ds, s)} & {ep}{'$^{*}$' if len(f_['epochs_budget']) > 1 else ''} & {pa} & ${lr:g}$ & {bn} & "
                 f"{d.get('random_resized_crop_scale_min', '--')} & {d.get('randaugment_num_ops', '--')}, {d.get('randaugment_magnitude', '--')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_hparams_students.tex", "\n".join(L))

    # methods: collapse identical settings across pairs
    def loss_of(ds, s, m):
        r = next(iter(A.RUNS.get((ds, s, m), {}).values()), None)
        return None if r is None else r["recipe"].get("method_loss")

    def pairs_label(prs):
        if len(prs) == len(PAIRS):
            return "all pairs"
        out_ = []
        for ds in A.DATASETS:
            ss = [s for d_, s in prs if d_ == ds]
            if len(ss) == len(A.STUDENTS):
                out_.append(f"{A.DS_SHORT[ds]} (all students)")
            elif ss:
                out_.append(f"{A.DS_SHORT[ds]}: " + ", ".join(A.SHORT[x] for x in ss))
        return "; ".join(out_)

    L = [r"\begin{tabular}{@{}lll@{}}", r"\toprule", r"Method & Loss setting & Pairs \\", r"\midrule"]
    fmt = {"ce_weight": r"CE$\times$%g", "distance_weight": r"dist.$\times$%g", "angle_weight": r"angle$\times$%g",
           "alpha": r"$\alpha{=}%g$", "beta": r"$\beta{=}%g$", "tau": r"$\tau{=}%g$"}
    for m in ["scratch", "kd", "rkd", "rbf_control", "pqk", "pqk_adaptive", "noent_control"]:
        inv = defaultdict(list)  # (setting, seed note) -> pairs
        for pr in PAIRS:
            by_seed = {sd_: r_["recipe"].get("method_loss") for sd_, r_ in A.RUNS.get((*pr, m), {}).items()}
            groups = defaultdict(list)
            for sd_, ml in sorted(by_seed.items()):
                if ml:
                    groups[json.dumps(ml, sort_keys=True)].append(sd_)
            for k, sds in groups.items():  # a cell whose seeds disagree gets one row per setting
                note = "" if len(groups) == 1 else f" (seed{'s' if len(sds) > 1 else ''} {', '.join(map(str, sds))})"
                inv[(k, note)].append(pr)
        for i, ((k, note), prs) in enumerate(sorted(inv.items(), key=lambda kv: (-len(kv[1]), kv[0][1]))):
            ml = json.loads(k)
            desc = ", ".join(fmt[f_] % ml[f_] for f_ in ("ce_weight", "alpha", "beta", "tau", "distance_weight", "angle_weight") if f_ in ml)
            if m in ("rbf_control", "pqk", "pqk_adaptive", "noent_control"):
                desc += r", $\gamma{=}1$ (10-epoch ramp)"
            L.append(f"{tt(m) if i == 0 else ''} & {desc} & {pairs_label(prs)}{note}{r'$^{\P}$' if note else ''} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_hparams_methods.tex", "\n".join(L))
    NUM["methods_loss"] = {f"{ds}/{s}/{m}": loss_of(ds, s, m) for ds, s in PAIRS for m in METH5 if loss_of(ds, s, m)}

    # kernel-method settings straight from the method configs
    if yaml:
        L = [r"\begin{tabular}{@{}llrrll@{}}", r"\toprule",
             r"Method & Kernel on the 8 angles & Qubits & Depth & $\lambda$ / bandwidth & Teacher projection \\", r"\midrule"]
        kc = {}
        for m, kdesc in [("rbf_control", "RBF"), ("pqk", "projected quantum (RY+CNOT)"),
                         ("pqk_adaptive", "projected quantum (RY+CNOT)"),
                         ("noent_control", r"classical map $[\sin4\theta,\cos4\theta]$"),
                         ("pqk_fit", "projected quantum (RY+CNOT)"), ("rbf_fit", "RBF")]:
            with open(os.path.join(A.REPO, "configs", "method", f"{m}.yaml"), encoding="utf-8") as f:
                c = yaml.safe_load(f)
            lam = c.get("lambda_", c.get("bandwidth", None))
            lam_s = "median heuristic" if lam is None else f"{lam:g}"
            proj = {"lda_pca": "LDA+PCA (fitted, frozen)"}.get(c.get("teacher_projection"), "random (frozen)")
            L.append(f"{tt(m)} & {kdesc} & {c.get('n_qubits', '--')} & {c.get('depth', '--')} & {lam_s} & {proj} \\\\")
            kc[m] = {"n_qubits": c.get("n_qubits"), "depth": c.get("depth"), "lambda_": c.get("lambda_"),
                     "bandwidth": c.get("bandwidth"), "teacher_projection": c.get("teacher_projection", "random"),
                     "backend": c.get("device_backend")}
        L += [r"\bottomrule", r"\end{tabular}"]
        write("tab_kernel_cfg.tex", "\n".join(L))
        NUM["kernel_cfg"] = kc

    # ------------------------------------------------------------------ teacher-kernel CKTA (scripts/paper_diagnostics.py)
    ck_path = os.path.join(A.GEN_DIR if hasattr(A, "GEN_DIR") else A.OUT, "ckta.json")
    if os.path.exists(ck_path) and os.path.getsize(ck_path):
        with open(ck_path, encoding="utf-8") as f:
            ck = json.load(f)
        names = {"angles (rbf_control)": "angles, RBF", "pqk 16 (RY+CNOT)": "PQK (RY+CNOT), 16 coords",
                 "no-CNOT 16 [sin4t,cos4t]": "no-CNOT, 16 coords"}
        L = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{TrashNet} & \multicolumn{2}{c}{RPS-25} \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Projection & Target kernel & val. & train batches & val. & train batches \\", r"\midrule"]
        T = {ds: ck[ds]["targets"] for ds in ("trashnet", "rps_25")}
        L.append(r"-- & raw teacher 2048-d, RBF & " + " & ".join(
            f"${T[ds]['raw teacher 2048-d'][sp]['mean']:.3f}$" for ds in ("trashnet", "rps_25") for sp in ("val", "train_batch")) + r" \\")
        for proj, plabel in (("random", "random (fixed)"), ("pca", "PCA (fitted)"), ("lda_pca", "LDA+PCA (fitted)")):
            L.append(r"\midrule" if proj else "")
            for j, (k, nm) in enumerate(names.items()):
                row = [f"${T[ds][f'{proj} | {k}'][sp]['mean']:.3f}$" for ds in ("trashnet", "rps_25") for sp in ("val", "train_batch")]
                L.append(f"{plabel if j == 0 else ''} & {nm} & " + " & ".join(row) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        write("tab_ckta.tex", "\n".join(x for x in L if x))
        NUM["ckta"] = {ds: {k: {sp: round(v[sp]["mean"], 4) for sp in v} for k, v in ck[ds]["targets"].items()}
                       for ds in ("trashnet", "rps_25")}

    # ------------------------------------------------------------------ rule #3 sensitivity (recipe-outlier seeds)
    # What changes if the run(s) trained under a different recipe than the rest of their pair are dropped?
    sens = {}
    for (ds, s, m), outl in A.RECIPE_OUTLIERS.items():
        allv, rec = A.seeds(ds, s, m), A.RUNS[(ds, s, m)]
        keep = {k: v for k, v in allv.items() if k not in outl}
        pk = A.seeds(ds, s, A.method_key(ds, s, "pqk"))
        entry = {"outlier_seeds": sorted(outl),
                 "outlier_recipe": {k: rec[min(outl)]["recipe"].get(k) for k in ("epochs_budget", "patience", "epochs_ran")},
                 "pair_recipe": {k: next(r["recipe"].get(k) for sd_, r in rec.items() if sd_ not in outl)
                                 for k in ("epochs_budget", "patience")},
                 "scores_by_seed": {str(k): round(v, 4) for k, v in sorted(allv.items())},
                 "mean_all": round(st.mean(allv.values()), 4), "std_all": round(st.stdev(allv.values()), 4),
                 "mean_matched": round(st.mean(keep.values()), 4), "std_matched": round(st.stdev(keep.values()), 4),
                 "n_matched": len(keep)}
        if m == "scratch":
            entry["pqk_minus_scratch_all"] = round(st.mean(pk.values()) - st.mean(allv.values()), 4)
            entry["pqk_minus_scratch_matched_seeds"] = round(
                st.mean(pk[k] for k in keep) - st.mean(keep.values()), 4)
            # fp32 variance count (pqk selected std < scratch std) with this pair's scratch std recomputed
            vc = var["fp32"]
            with_, without_ = 0, 0
            for ds2, s2 in PAIRS:
                ps, ss = vc[f"{ds2}/{s2}"]["pqk"], vc[f"{ds2}/{s2}"]["scratch"]
                with_ += ps < ss
                without_ += ps < (entry["std_matched"] if (ds2, s2) == (ds, s) else ss)
            entry["fp32_pairs_pqk_std_below_scratch_with"] = int(with_)
            entry["fp32_pairs_pqk_std_below_scratch_without"] = int(without_)
        sens[f"{ds}/{s}/{m}"] = entry
    NUM["rule3_sensitivity"] = sens

    # cells whose seeds disagree on the loss settings (e.g. the KD temperature): what changes if the minority seeds are dropped
    loss_sens = {}
    for (ds_, s_, m_), outl in A.LOSS_OUTLIERS.items():
        runs_ = A.RUNS[(ds_, s_, m_)]
        allv = A.seeds(ds_, s_, m_)
        keep = {k: v for k, v in allv.items() if k not in outl}
        pk = A.seeds(ds_, s_, A.method_key(ds_, s_, "pqk"))
        e = {"tau_by_seed": {str(k): runs_[k]["recipe"]["method_loss"].get("tau") for k in sorted(runs_)},
             "scores_by_seed": {str(k): round(v, 4) for k, v in sorted(allv.items())},
             "minority_seeds": sorted(outl), "mean_all": round(st.mean(allv.values()), 4), "std_all": round(st.stdev(allv.values()), 4),
             "mean_majority": round(st.mean(keep.values()), 4), "std_majority": round(st.stdev(keep.values()), 4), "n_majority": len(keep)}
        if m_ == "kd":
            e["pqk_minus_kd_all"] = round(st.mean(pk.values()) - st.mean(allv.values()), 4)
            e["pqk_minus_kd_majority_seeds"] = round(st.mean(pk[k] for k in keep) - st.mean(keep.values()), 4)
            e["pqk_minus_kd_minority_seed"] = {str(k): round(pk[k] - allv[k], 4) for k in sorted(outl)}
            ahead = 0
            for ds2, s2 in PAIRS:
                q, k2 = A.seeds(ds2, s2, A.method_key(ds2, s2, "pqk")), A.seeds(ds2, s2, "kd")
                ks = [x for x in q if x in k2 and not ((ds2, s2) == (ds_, s_) and x in outl)]
                ahead += st.mean(q[x] - k2[x] for x in ks) > 0
            e["fp32_pairs_pqk_ahead_of_kd_with"] = NUM["pqk_vs_kd"]["ahead"]["fp32"][0]
            e["fp32_pairs_pqk_ahead_of_kd_without"] = int(ahead)
        loss_sens[f"{ds_}/{s_}/{m_}"] = e
    NUM["loss_mixed_sensitivity"] = loss_sens

    # ------------------------------------------------------------------ rule #3 audit table (one row per seed of each outlier cell)
    L = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Run & Seed & Budget & Patience & Epochs run & Macro-F1 \\", r"\midrule"]
    for key, e in sens.items():
        ds_, s_, m_ = key.split("/")
        for sd_ in sorted(A.RUNS[(ds_, s_, m_)]):
            rc = A.RUNS[(ds_, s_, m_)][sd_]["recipe"]
            flag = r"$^{\S}$" if sd_ in A.RECIPE_OUTLIERS[(ds_, s_, m_)] else ""
            L.append(f"{A.DS_SHORT[ds_]}/{A.SHORT[s_]}, {tt(m_)} & {sd_}{flag} & {rc['epochs_budget']} & {rc['patience']} & "
                     f"{rc['epochs_ran']} & ${A.RUNS[(ds_, s_, m_)][sd_]['macro_f1']:.3f}$ \\\\")
        L += [r"\midrule",
              f"all three seeds (as reported) & & & & & ${e['mean_all']:.3f}\\pm{e['std_all']:.3f}$ \\\\",
              f"matched seeds only ($n={e['n_matched']}$) & & & & & ${e['mean_matched']:.3f}\\pm{e['std_matched']:.3f}$ \\\\"]
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_rule3.tex", "\n".join(L))

    # ------------------------------------------------------------------ pqk vs rkd (QRKD's claimed comparison)
    L = [r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
         r"Dataset & Student & \texttt{rkd} & $\Delta$ \texttt{pqk} $-$ \texttt{rkd} \\", r"\midrule"]
    pr = {}
    for ds, s in PAIRS:
        rk = A.seeds(ds, s, "rkd")
        if len(rk) < 3:
            L.append(f"{pair_name(ds, s)} & -- & -- \\\\")
            continue
        d = paired(fp32(ds, s, "pqk"), rk)
        L.append(f"{pair_name(ds, s)} & {pm(list(rk.values()))} & {dcell(d)} \\\\")
        pr[f"{ds}/{s}"] = {"diff": round(st.mean(d), 4), "wins": sum(x > 0 for x in d)}
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_pqk_rkd.tex", "\n".join(L))
    NUM["pqk_vs_rkd"] = {"pairs": pr, "ahead": [sum(v["diff"] > 0 for v in pr.values()), len(pr)],
                         "max_lead": round(max(v["diff"] for v in pr.values()), 4)}
    NUM["rbf_beats_scratch_pairs"] = [sum(st.mean(A.seeds(ds, s, "rbf_control").values()) > st.mean(A.seeds(ds, s, "scratch").values())
                                          for ds, s in PAIRS), len(PAIRS)]

    # ------------------------------------------------------------------ kernel health (mean off-diagonal K_ij, Trap #3 / #13)
    health = defaultdict(lambda: {"s": [], "t": []})
    for r in A.DATA["students"]:
        if "kernel_offdiag_student" in r and r["method"] in ("rbf_control", "pqk", "pqk_adaptive", "noent_control", "pqk_fit", "rbf_fit"):
            health[r["method"]]["s"].append(r["kernel_offdiag_student"])
            health[r["method"]]["t"].append(r["kernel_offdiag_teacher"])
    L = [r"\begin{tabular}{@{}lrcc@{}}", r"\toprule",
         r"Method & Runs & Student, mean [min, max] & Teacher, mean [min, max] \\", r"\midrule"]
    for m in ("rbf_control", "noent_control", "pqk", "pqk_adaptive", "rbf_fit", "pqk_fit"):
        h = health[m]
        L.append(f"{tt(m)} & {len(h['s'])} & ${st.mean(h['s']):.3f}$ [{min(h['s']):.3f}, {max(h['s']):.3f}] & "
                 f"${st.mean(h['t']):.3f}$ [{min(h['t']):.3f}, {max(h['t']):.3f}] \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_kernel_health.tex", "\n".join(L))
    arch = {}
    for p in glob.glob(os.path.join(A.REPO, "archive_prefix_kernel", "*", "metrics.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if "kernel_offdiag_student" in r:
            arch[r["run_id"]] = (r["kernel_offdiag_student"], r["kernel_offdiag_teacher"], r["method"])
    collapsed = max((v for v in arch.values() if v[2] == "pqk"), key=lambda v: v[0])
    classical = [v for v in arch.values() if v[2] == "rbf_control"]
    NUM["kernel_health"] = {"post_fix": {m: {"n": len(h["s"]), "student_mean": round(st.mean(h["s"]), 3),
                                              "student_min": round(min(h["s"]), 3), "student_max": round(max(h["s"]), 3),
                                              "teacher_min": round(min(h["t"]), 3), "teacher_max": round(max(h["t"]), 3)}
                                         for m, h in health.items()},
                            "pre_fix_collapse": {"student": round(collapsed[0], 4), "teacher": round(collapsed[1], 4),
                                                 "classical_twin_student_range": [round(min(v[0] for v in classical), 3),
                                                                                  round(max(v[0] for v in classical), 3)],
                                                 "n_archive_runs": len(arch)}}

    # ------------------------------------------------------------------ paired t-tests (uncorrected, n = 3 per pair)
    from scipy import stats as _stats

    def pval(x, y):
        ks = sorted(set(x) & set(y))
        return float(_stats.ttest_rel([x[k] for k in ks], [y[k] for k in ks]).pvalue) if len(ks) >= 3 else None

    tests = {}
    for other in ("kd", "rbf_control", "rkd", "scratch"):
        tests[other] = {f"{ds}/{s}": pval(fp32(ds, s, "pqk"), A.seeds(ds, s, other)) for ds, s in PAIRS}
    tests["int8_vs_kd"] = {k: {f"{ds}/{s}": pval(int8(ds, s, "pqk", k), int8(ds, s, "kd", k)) for ds, s in EDGE_PAIRS}
                           for k, _ in NUMERICS}
    # the two pqk calibrations against the twin, separately (adaptive only where it has 3 seeds)
    tests["rbf_fixed"] = {f"{ds}/{s}": pval(A.seeds(ds, s, "pqk"), A.seeds(ds, s, "rbf_control")) for ds, s in PAIRS}
    tests["rbf_adaptive"] = {f"{ds}/{s}": pval(A.seeds(ds, s, "pqk_adaptive"), A.seeds(ds, s, "rbf_control")) for ds, s in PAIRS}
    minp = {o: min(v for v in tests[o].values() if v is not None)
            for o in ("kd", "rbf_control", "rkd", "scratch", "rbf_fixed", "rbf_adaptive")}
    minp["int8_vs_kd"] = min(v for d in tests["int8_vs_kd"].values() for v in d.values() if v is not None)
    NUM["paired_tests"] = {"note": "two-sided paired t-test over 3 seeds, uncorrected for multiple comparisons",
                           "p": {o: ({k: (None if v is None else round(v, 3)) for k, v in tests[o].items()}) for o in
                                 ("kd", "rbf_control", "rkd", "scratch", "rbf_fixed", "rbf_adaptive")},
                           "int8_vs_kd": {k: {p: round(v, 3) for p, v in d.items()} for k, d in tests["int8_vs_kd"].items()},
                           "min_p": {k: round(v, 3) for k, v in minp.items()}}

    # which side the most significant adaptive-lambda-vs-twin difference favours (it is the one nominal p < 0.05)
    _k = min((k for k, v in tests["rbf_adaptive"].items() if v is not None), key=lambda k: tests["rbf_adaptive"][k])
    _ds, _s = _k.split("/")
    _d = st.mean(paired(A.seeds(_ds, _s, "pqk_adaptive"), A.seeds(_ds, _s, "rbf_control")))
    adapt_min_dir = "the classical twin" if _d < 0 else "PQK"
    NUM["paired_tests"]["adaptive_vs_twin_most_significant"] = {"pair": _k, "diff": round(_d, 4),
                                                                "p": round(tests["rbf_adaptive"][_k], 3), "favours": adapt_min_dir}

    # ------------------------------------------------------------------ macros for prose (letters-only names)
    # paired test on the INT8-minus-fp32 change, pqk vs kd (12 model pairs per numerics)
    qd = {}
    for k, _ in NUMERICS:
        dp, dk = [], []
        for ds, s in EDGE_PAIRS:
            P, K = A.numerics_scores(ds, s, A.exported_variant(ds, s)), A.numerics_scores(ds, s, "kd")
            for sd_ in sorted(P[k]):
                dp.append(P[k][sd_] - P["fp32"][sd_])
                dk.append(K[k][sd_] - K["fp32"][sd_])
        qd[k] = {"diff": st.mean(x - y for x, y in zip(dp, dk)), "p": float(_stats.ttest_rel(dp, dk).pvalue), "n": len(dp)}
    NUM["quant_delta_pqk_minus_kd"] = {k: {"diff": round(v["diff"], 4), "p": round(v["p"], 3), "n": v["n"]} for k, v in qd.items()}
    xs = []
    for ds, s in EDGE_PAIRS:
        for m in ("scratch", "kd", "rkd", "rbf_control", "pqk", "pqk_adaptive"):
            sc = A.numerics_scores(ds, s, m)
            if sc["xnnpack"] and sc["pc"]:
                xs.append(st.mean(sc["xnnpack"].values()) - st.mean(sc["pc"].values()))
    NUM["xnnpack_minus_pc_f1_range"] = [round(min(xs), 4), round(max(xs), 4)]
    ivs = NUM["variance"]["int8_summary"]["pc"]["pooled"]

    # variance sensitivities: same five pairs, mismatched seed removed, and the other metric (top-1)
    five = [(ds, s) for ds, s in PAIRS if len(A.seeds(ds, s, "rkd")) == 3]

    def pooled_over(method, prs, fix=False, key="macro_f1"):
        v = []
        for ds, s in prs:
            k = A.method_key(ds, s, "pqk") if method == "pqk" else method
            x = A.seeds(ds, s, k, key)
            if fix and method == "scratch" and (ds, s) in {tuple(q.split("/")[:2]) for q in sens}:
                x = {sd_: val for sd_, val in x.items() if sd_ not in A.RECIPE_OUTLIERS[(ds, s, "scratch")]}
            v.append(st.stdev(x.values()) ** 2)
        return math.sqrt(st.mean(v))

    def top1_counts(fix=False):
        lo_s = lo_r = 0
        for ds, s in PAIRS:
            q = st.stdev(A.seeds(ds, s, A.method_key(ds, s, "pqk"), "top1_accuracy").values())
            sc = A.seeds(ds, s, "scratch", "top1_accuracy")
            if fix:
                sc = {k: v for k, v in sc.items() if k not in A.RECIPE_OUTLIERS.get((ds, s, "scratch"), set())}
            lo_s += q < st.stdev(sc.values())
            lo_r += q < st.stdev(A.seeds(ds, s, "rbf_control", "top1_accuracy").values())
        return int(lo_s), int(lo_r)

    t1, t1x = top1_counts(), top1_counts(fix=True)
    NUM["variance_sensitivity"] = {
        "five_pair_pooled": {m: round(pooled_over(m, five), 4) for m in ("pqk", "rkd", "scratch", "rbf_control", "kd")},
        "six_pair_pooled_scratch_without_mismatched_seed": round(pooled_over("scratch", PAIRS, fix=True), 4),
        "rkd_below_pqk_on_five_pairs": sum(st.stdev(A.seeds(ds, s, "rkd").values()) <
                                           st.stdev(A.seeds(ds, s, A.method_key(ds, s, "pqk")).values()) for ds, s in five),
        "top1_pqk_below_scratch_rbf": list(t1), "top1_pqk_below_scratch_rbf_without_mismatched_seed": list(t1x)}
    ed_ = [r for r in A.EDGE if r["precision"] == "int8" and r["threads"] == 4 and "parity" in r]
    dis = {k: [1 - r["parity"]["argmax_agreement"] for r in ed_ if r["resolver"] == res]
           for k, res in (("xnnpack", "auto"), ("builtin", "no_default_delegates"))}
    NUM["device_disagreement_per_model"] = {k: {"min": round(min(v), 4), "max": round(max(v), 4), "mean": round(st.mean(v), 4), "n": len(v)}
                                            for k, v in dis.items()}
    dm = NUM["device_disagreement_per_model"]
    mb = {s: st.mean(r["tflite"]["size_kb"] * 1024 / 1e6 for r in A.DATA["deploy"] if r["student"] == s) for s in A.EDGE_STUDENTS}

    g, vsum, ia = NUM["g2"], NUM["variance"]["fp32_summary"], NUM["int8_all_models"]
    kdfp = NUM["pqk_vs_kd"]["fp32"]
    ed = NUM["edge"]
    kd_gain = {p: st.mean(A.seeds("trashnet", p, "kd").values()) - st.mean(A.seeds("trashnet", p, "scratch").values())
               for p in A.EDGE_STUDENTS}
    sens_ex = NUM["rule3_sensitivity"]["rps_25/lenet5/scratch"]
    M = {
        "NStudentRuns": NUM["counts"]["student_runs"], "NTeachers": NUM["counts"]["teachers"],
        "NGridRuns": sum(len(A.seeds(ds, s, A.method_key(ds, s, m))) for ds, s in PAIRS for m in METH5),
        "NIntModels": NUM["counts"]["int8_exports"], "NSeeds": 3, "NPairs": len(PAIRS), "NExported": len(EDGE_PAIRS),
        "GtwoFixedWins": g["fixed_pair_wins"], "GtwoAdaptWins": g["adaptive_pair_wins"], "GtwoAdaptPairs": g["n_adaptive_pairs"],
        "KdAheadFp": NUM["pqk_vs_kd"]["ahead"]["fp32"][0], "KdAheadPc": NUM["pqk_vs_kd"]["ahead"]["pc"][0],
        "KdAheadBuiltin": NUM["pqk_vs_kd"]["ahead"]["builtin"][0], "KdAheadXnn": NUM["pqk_vs_kd"]["ahead"]["xnnpack"][0],
        "KdBehindTnA": f"{-kdfp['trashnet/mobilenetv2_035']['diff']:.3f}",
        "KdBehindTnB": f"{-kdfp['trashnet/mobilenetv3_small']['diff']:.3f}",
        "KdGainTnA": f"{kd_gain['mobilenetv2_035']:.3f}", "KdGainTnB": f"{kd_gain['mobilenetv3_small']:.3f}",
        "RkdAhead": NUM["pqk_vs_rkd"]["ahead"][0], "RkdPairs": NUM["pqk_vs_rkd"]["ahead"][1],
        "RkdMaxLead": f"{NUM['pqk_vs_rkd']['max_lead']:.3f}",
        "VarBelowScratch": vsum["selected_lower_than"]["scratch"][0], "VarBelowRbf": vsum["selected_lower_than"]["rbf_control"][0],
        "VarBelowScratchEx": sens_ex["fp32_pairs_pqk_std_below_scratch_without"],
        "PooledPqk": f"{pooled(rows32, 'pqk'):.3f}", "PooledRkd": f"{pooled(rows32, 'rkd'):.3f}",
        "PooledScratch": f"{pooled(rows32, 'scratch'):.3f}", "PooledRbf": f"{pooled(rows32, 'rbf_control'):.3f}",
        "SignP": f"{vsum['sign_test_p_4of6']:.2f}",
        "MinPKd": f"{minp['kd']:.2f}", "MinPRbf": f"{minp['rbf_control']:.2f}", "MinPKdInt": f"{minp['int8_vs_kd']:.2f}",
        "MinPScratch": f"{minp['scratch']:.2f}", "MinPRbfFixed": f"{minp['rbf_fixed']:.2f}",
        "MinPRbfAdapt": f"{minp['rbf_adaptive']:.2f}",
        "MinPRbfAdaptFavours": adapt_min_dir, "RbfBeatsScratch": NUM["rbf_beats_scratch_pairs"][0],
        "LatMnvTwo": f"{ed['trashnet/mobilenetv2_035']['p50_ms']:.1f}", "LatMnvThree": f"{ed['trashnet/mobilenetv3_small']['p50_ms']:.1f}",
        "MbMnvTwo": f"{mb['mobilenetv2_035']:.2f}", "MbMnvThree": f"{mb['mobilenetv3_small']:.2f}",
        "PooledPqkFive": f"{NUM['variance_sensitivity']['five_pair_pooled']['pqk']:.3f}",
        "PooledRkdFive": f"{NUM['variance_sensitivity']['five_pair_pooled']['rkd']:.3f}",
        "PooledScratchFive": f"{NUM['variance_sensitivity']['five_pair_pooled']['scratch']:.3f}",
        "PooledScratchEx": f"{NUM['variance_sensitivity']['six_pair_pooled_scratch_without_mismatched_seed']:.3f}",
        "VarTopBelowScratch": t1[0], "VarTopBelowRbf": t1[1], "VarTopBelowScratchEx": t1x[0],
        "XnnDisMin": f"{dm['xnnpack']['min'] * 100:.1f}", "XnnDisMax": f"{dm['xnnpack']['max'] * 100:.1f}",
        "XnnDisMean": f"{dm['xnnpack']['mean'] * 100:.1f}", "BuiltinDisMax": f"{dm['builtin']['max'] * 100:.1f}",
        "BuiltinDisMean": f"{dm['builtin']['mean'] * 100:.1f}",
        "KdAheadFpEx": NUM["loss_mixed_sensitivity"]["rps_25/lenet5/kd"]["fp32_pairs_pqk_ahead_of_kd_without"],
        "IntDeltaMean": f"{ia['mean']:+.3f}", "IntDeltaStd": f"{ia['std']:.3f}", "IntGained": ia["gained"],
        "TeacherTn": f"{NUM['teachers']['trashnet']['macro_f1']:.3f}", "TeacherRps": f"{NUM['teachers']['rps_25']['macro_f1']:.3f}",
        "CostKeptHours": f"{NUM['cost']['kept_hours']:.0f}", "CostDiscardedHours": f"{NUM['cost']['discarded_hours']:.0f}",
        "CostPqkHours": f"{NUM['cost']['pqk_family_kept_hours']:.0f}", "CostPqkRuns": NUM["cost"]["pqk_runs"],
        "QuantPPc": f"{qd['pc']['p']:.2f}", "QuantPBuiltin": f"{qd['builtin']['p']:.2f}", "QuantPXnn": f"{qd['xnnpack']['p']:.2f}",
        "XnnVsPcMin": f"{min(xs):+.3f}", "XnnVsPcMax": f"{max(xs):+.3f}",
        "IntPooledPqk": f"{ivs['pqk']:.3f}", "IntPooledScratch": f"{ivs['scratch']:.3f}", "IntPooledRbf": f"{ivs['rbf_control']:.3f}",
        "CollapseStudent": f"{NUM['kernel_health']['pre_fix_collapse']['student']:.3f}",
        "CollapseTeacher": f"{NUM['kernel_health']['pre_fix_collapse']['teacher']:.3f}",
        "OffdiagFixedMin": f"{NUM['kernel_health']['post_fix']['pqk']['student_min']:.2f}",
        "OffdiagFixedMax": f"{NUM['kernel_health']['post_fix']['pqk']['student_max']:.2f}",
        "OffdiagOthersMin": f"{min(NUM['kernel_health']['post_fix'][m]['student_min'] for m in ('rbf_control', 'noent_control', 'pqk_adaptive', 'rbf_fit', 'pqk_fit')):.2f}",
        "OffdiagOthersMax": f"{max(NUM['kernel_health']['post_fix'][m]['student_max'] for m in ('rbf_control', 'noent_control', 'pqk_adaptive', 'rbf_fit', 'pqk_fit')):.2f}",
    }
    with open(os.path.join(A.OUT, "macros.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("% generated by scripts/aggregate.py — do not edit\n")
        for k, v in M.items():
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("wrote paper/generated/macros.tex")
    NUM["macros"] = {k: str(v) for k, v in M.items()}
