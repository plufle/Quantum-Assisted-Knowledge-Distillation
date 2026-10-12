"""Paper figures (static vector PDFs), all generated from repo data — no generative AI, no manual drawing.

    python scripts/paper_figures.py [--only name,name]     # -> paper/figures/*.pdf

Needs paper/generated/numbers.json (run scripts/aggregate.py first) and, for two figures,
paper/generated/{ckta,gradient_variance}.json (scripts/paper_diagnostics.py).

Colour follows the dataviz skill: blue = the quantum-kernel arm, orange = its classical counterpart,
aqua = scratch, gray = everything else. Identity is also carried by marker shape / position / direct
labels, never by colour alone. Raw per-seed points are always drawn beside means (n = 3).
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate as agg  # noqa: E402  (module-level load only; main() is guarded)

GEN = os.path.join(agg.REPO, "paper", "generated")
FIG = os.path.join(agg.REPO, "paper", "figures")
BLUE, ORANGE, AQUA, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985", "#0b0b0b", "#52514e", "#e4e3df"
LIGHT_BLUE, PAPER = "#dbe8f8", "#f6f5f2"

plt.rcParams.update({
    "pdf.fonttype": 42, "font.family": "serif", "font.size": 8.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "mathtext.fontset": "dejavuserif",
})
PAIRS = [(ds, s) for ds in agg.DATASETS for s in agg.STUDENTS]
EDGE_PAIRS = [(ds, s) for ds in agg.DATASETS for s in agg.EDGE_STUDENTS]
PNG_DIR = os.environ.get("QAKD_FIG_PNG")  # optional: also write PNGs here for eyeballing


def numj():
    with open(os.path.join(GEN, "numbers.json"), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    if PNG_DIR:
        fig.savefig(os.path.join(PNG_DIR, f"{name}.png"), dpi=130)
    plt.close(fig)
    print(f"wrote paper/figures/{name}.pdf")


FSHORT = {"mobilenetv2_035": "MNv2-0.35", "mobilenetv3_small": "MNv3-S", "lenet5": "LeNet-5"}  # compact names for axis labels


def label(ds, s):
    return f"{agg.DS_SHORT[ds]} / {FSHORT[s]}"


# ----------------------------------------------------------------------------- results figures
def fig_g2_paired():
    """Per-seed paired difference pqk - rbf_control (test macro-F1), every pair, both lambda calibrations."""
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    for i, (ds, stu) in enumerate(PAIRS):
        y = len(PAIRS) - 1 - i
        rbf = agg.seeds(ds, stu, "rbf_control")
        for key, color, marker, dy in (("pqk", BLUE, "o", 0.16), ("pqk_adaptive", ORANGE, "D", -0.16)):
            q = agg.seeds(ds, stu, key)
            if len(q) < 3:
                continue
            d = agg.paired(q, rbf)
            ax.scatter(d, [y + dy] * len(d), s=11, color=color, alpha=0.45, linewidths=0, zorder=3)
            ax.scatter([st.mean(d)], [y + dy], s=42, color=color, marker=marker, edgecolors="white", linewidths=0.8, zorder=4)
    ax.axvline(0, color=INK, linewidth=0.8, zorder=2)
    ax.set_yticks(range(len(PAIRS)))
    ax.set_yticklabels([label(*p) for p in reversed(PAIRS)])
    ax.set_xlabel(r"test macro-F1: $\mathtt{pqk}-\mathtt{rbf\_control}$ (paired by seed)")
    ax.set_ylim(-0.6, len(PAIRS) - 0.4)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(0.99, 0.02, "favours pqk →", transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=7.5)
    ax.text(0.01, 0.02, "← favours rbf_control", transform=ax.transAxes, ha="left", va="bottom", color=MUTED, fontsize=7.5)
    h = [Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=6, label=r"$\lambda=1$ (pre-specified)"),
         Line2D([], [], marker="D", color=ORANGE, linestyle="", markersize=5.5, label=r"adaptive $\lambda$ (post hoc; LeNet-5: $n{=}1$, omitted)")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "g2_paired")


def fig_methods_seeds():
    """Per-seed test macro-F1, five methods, four mobile-student pairs (2x2 small multiples)."""
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.4), sharex=True)
    for ax, (ds, stu) in zip(axes.flat, EDGE_PAIRS):
        ax.axhline(st.mean(agg.seeds(ds, stu, "scratch").values()), color=GRAY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        for x, m in enumerate(agg.METHODS):
            v = list(agg.seeds(ds, stu, agg.method_key(ds, stu, m)).values())
            c = BLUE if m == "pqk" else GRAY
            ax.scatter([x + (k - 1) * 0.07 for k in range(len(v))], v, s=13, color=c, alpha=0.5, linewidths=0, zorder=3)
            ax.hlines(st.mean(v), x - 0.22, x + 0.22, color=BLUE if m == "pqk" else INK, linewidth=1.6, zorder=4)
        ax.set_title(label(ds, stu), fontsize=8.5, loc="left")
        ax.set_xticks(range(len(agg.METHODS)))
        ax.set_xticklabels(agg.METHODS, rotation=30, ha="right", fontsize=7.5)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("test macro-F1")
    fig.text(0.99, 0.005, "dashed = scratch mean; bar = 3-seed mean; dots = seeds; pqk at its validation-selected λ",
             ha="right", va="bottom", color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, "methods_seeds")


def fig_pqk_vs_kd():
    """pqk - kd, paired by seed: fp32 for all six pairs, INT8 under three numerics for the four exported pairs."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.2), sharex=True, gridspec_kw={"width_ratios": [6, 4.4]})
    for i, (ds, s) in enumerate(PAIRS):
        y = len(PAIRS) - 1 - i
        d = agg.paired(agg.seeds(ds, s, agg.method_key(ds, s, "pqk")), agg.seeds(ds, s, "kd"))
        a1.scatter(d, [y] * 3, s=11, color=BLUE, alpha=0.45, linewidths=0, zorder=3)
        a1.scatter([st.mean(d)], [y], s=44, color=BLUE, edgecolors="white", linewidths=0.8, zorder=4)
    a1.set_yticks(range(len(PAIRS)))
    a1.set_yticklabels([label(*p) for p in reversed(PAIRS)], fontsize=7.5)
    a1.set_ylim(-0.6, len(PAIRS) - 0.4)
    a1.set_title("fp32", fontsize=8.5, loc="left")
    marks = [("pc", "o", "PC (builtin)", 0.22), ("builtin", "s", "Pi builtin", 0.0), ("xnnpack", "D", "Pi XNNPACK", -0.22)]
    for i, (ds, s) in enumerate(EDGE_PAIRS):
        y = len(EDGE_PAIRS) - 1 - i
        for k, mk, _, dy in marks:
            P = agg.numerics_scores(ds, s, agg.exported_variant(ds, s))[k]
            K = agg.numerics_scores(ds, s, "kd")[k]
            d = agg.paired(P, K)
            a2.scatter(d, [y + dy] * 3, s=8, color=BLUE, alpha=0.4, linewidths=0, marker=mk, zorder=3)
            a2.scatter([st.mean(d)], [y + dy], s=34, color=BLUE, marker=mk, edgecolors="white", linewidths=0.7, zorder=4)
    a2.set_yticks(range(len(EDGE_PAIRS)))
    a2.set_yticklabels([label(*p) for p in reversed(EDGE_PAIRS)], fontsize=7.5)
    a2.set_ylim(-0.6, len(EDGE_PAIRS) - 0.4)
    a2.set_title("INT8", fontsize=8.5, loc="left")
    for ax in (a1, a2):
        ax.axvline(0, color=INK, linewidth=0.8, zorder=2)
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel(r"test macro-F1: $\mathtt{pqk}-\mathtt{kd}$")
    h = [Line2D([], [], marker=mk, color=BLUE, linestyle="", markersize=5.5, label=lab) for _, mk, lab, _ in marks]
    a1.scatter([], [], s=44, color=BLUE, label="3-seed mean")
    a1.scatter([], [], s=11, color=BLUE, alpha=0.45, label="single seed")
    h = [a1.collections[-2], a1.collections[-1]] + h
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=7, handletextpad=0.2, columnspacing=1.0,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "pqk_vs_kd")


def fig_variance():
    """Seed-to-seed std of test macro-F1 per method per pair, fp32 (6 pairs) and INT8 on PC (4 pairs)."""
    V = numj()["variance"]
    style = {"pqk": (BLUE, "o", 46, 5, r"$\mathtt{pqk}$"), "scratch": (AQUA, "s", 30, 3, r"$\mathtt{scratch}$"),
             "rbf_control": (ORANGE, "D", 28, 3, r"$\mathtt{rbf\_control}$"), "kd": (GRAY, "^", 28, 2, r"$\mathtt{kd}$"),
             "rkd": (GRAY, "v", 28, 2, r"$\mathtt{rkd}$")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.3), gridspec_kw={"width_ratios": [6, 4.4]})
    for ax, data, pairs, title in ((a1, V["fp32"], PAIRS, "fp32"), (a2, V["int8"]["pc"], EDGE_PAIRS, "INT8 (PC)")):
        for i, (ds, s) in enumerate(pairs):
            y = len(pairs) - 1 - i
            row = data[f"{ds}/{s}"]
            for m, (c, mk, sz, z, _) in style.items():
                v = row.get(m)
                if v is None:
                    continue
                hollow = m in ("kd", "rkd")
                ax.scatter([v], [y], s=sz, color="white" if hollow else c, edgecolors=c, linewidths=1.0 if hollow else 0.7,
                           marker=mk, zorder=z)
        ax.set_xscale("log")
        ax.set_xlim(0.0018, 0.35)
        ax.set_xticks([0.003, 0.03, 0.3])
        ax.set_xticklabels(["0.003", "0.03", "0.3"])
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([label(*p) for p in reversed(pairs)], fontsize=7.5)
        ax.set_ylim(-0.6, len(pairs) - 0.4)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel("seed std, test macro-F1 (log)")
        ax.grid(axis="x", color=GRID, linewidth=0.5, which="major")
        ax.set_axisbelow(True)
    h = [Line2D([], [], marker=mk, color=c if m not in ("kd", "rkd") else GRAY, markerfacecolor="white" if m in ("kd", "rkd") else c,
                linestyle="", markersize=5.5, label=lab) for m, (c, mk, _, _, lab) in style.items()]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0), handletextpad=0.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "variance")


def fig_fitted():
    """(a) fitted - random teacher projection, paired validation macro-F1; (b,c) CKTA of the target kernel on
    validation vs on 128-image training batches (the label-saturation mechanism)."""
    ck_path = os.path.join(GEN, "ckta.json")
    ck = json.load(open(ck_path, encoding="utf-8")) if os.path.exists(ck_path) and os.path.getsize(ck_path) else None
    fig = plt.figure(figsize=(6.5, 3.4))
    gs = fig.add_gridspec(1, 3 if ck else 1, width_ratios=[1.5, 1, 1] if ck else [1])
    a = fig.add_subplot(gs[0])
    rows = [(ds, s, arm) for ds, s in EDGE_PAIRS for arm in ("pqk", "rbf")]
    for i, (ds, s, arm) in enumerate(rows):
        y = len(rows) - 1 - i - (0.35 if i % 2 == 0 else 0.0)
        base, fit, c, mk = (("pqk_adaptive", "pqk_fit", BLUE, "o") if arm == "pqk" else ("rbf_control", "rbf_fit", ORANGE, "D"))
        d = agg.paired(agg.seeds(ds, s, fit, "best_val_macro_f1"), agg.seeds(ds, s, base, "best_val_macro_f1"))
        a.scatter(d, [y] * 3, s=10, color=c, alpha=0.45, linewidths=0, zorder=3)
        a.scatter([st.mean(d)], [y], s=40, color=c, marker=mk, edgecolors="white", linewidths=0.8, zorder=4)
    ys = [len(rows) - 1 - i - (0.35 if i % 2 == 0 else 0.0) for i in range(len(rows))]
    a.set_yticks([(ys[2 * k] + ys[2 * k + 1]) / 2 for k in range(len(EDGE_PAIRS))])
    a.set_yticklabels([label(*p) for p in EDGE_PAIRS], fontsize=7.5)
    a.axvline(0, color=INK, linewidth=0.8)
    a.set_xlabel("validation macro-F1: fitted $-$ random")
    a.set_title("(a) fitted vs random projection", fontsize=8.5, loc="left")
    a.grid(axis="x", color=GRID, linewidth=0.5)
    a.set_axisbelow(True)
    h = [Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=5.5, label="pqk"),
         Line2D([], [], marker="D", color=ORANGE, linestyle="", markersize=5, label="rbf_control")]
    fig.legend(handles=h, loc="lower left", ncol=2, frameon=False, fontsize=7, handletextpad=0.1, bbox_to_anchor=(0.01, 0.0))
    if ck:
        targets = [("random | angles (rbf_control)", "random", "RBF"), ("random | pqk 16 (RY+CNOT)", "random", "PQK"),
                   ("lda_pca | angles (rbf_control)", "fitted", "RBF"), ("lda_pca | pqk 16 (RY+CNOT)", "fitted", "PQK")]
        for j, ds in enumerate(("trashnet", "rps_25")):
            ax = fig.add_subplot(gs[1 + j], sharey=None)
            T = ck[ds]["targets"]
            for k, (key, proj, kern) in enumerate(targets):
                y = len(targets) - 1 - k - (0.3 if k >= 2 else 0)
                c = BLUE if kern == "PQK" else ORANGE
                v, t = T[key]["val"]["mean"], T[key]["train_batch"]["mean"]
                ax.plot([v, t], [y, y], color=c, linewidth=1.2, alpha=0.6, zorder=2)
                ax.scatter([v], [y], s=34, facecolor="white", edgecolors=c, linewidths=1.2, zorder=4)
                ax.scatter([t], [y], s=34, color=c, zorder=5)
            ax.set_xlim(0.1, 1.02)
            ax.set_yticks([len(targets) - 1 - k - (0.3 if k >= 2 else 0) for k in range(len(targets))])
            ax.set_yticklabels([f"{p} {kk}" for _, p, kk in targets] if j == 0 else [], fontsize=7)
            ax.set_xlabel("CKTA with label kernel")
            ax.set_title(f"({'bc'[j]}) {agg.DS_SHORT[ds]}", fontsize=8.5, loc="left")
            ax.grid(axis="x", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
        h2 = [Line2D([], [], marker="o", markerfacecolor="white", color=MUTED, linestyle="", markersize=5.5, label="validation"),
              Line2D([], [], marker="o", color=MUTED, linestyle="", markersize=5.5, label="128-image training batches")]
        fig.legend(handles=h2, loc="lower right", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.99, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fitted_projection")


def fig_gradients():
    """Kernel-term / CE gradient-norm ratio over training (first batch of each epoch, mean over seeds)."""
    arms = [("noent_control", ORANGE, "-", "no-CNOT, random target"), ("pqk_adaptive", BLUE, "-", "pqk, random target"),
            ("pqk_fit", BLUE, (0, (4, 2)), "pqk, fitted target"), ("rbf_fit", ORANGE, (0, (4, 2)), "RBF, fitted target")]
    raw = agg.DATA["students"]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2), sharex=True, sharey=True)
    for ax, (ds, s) in zip(axes.flat, EDGE_PAIRS):
        for m, c, ls, lab in arms:
            logs = [r["grad_norms"] for r in raw if (r["dataset"], r["student"], r["method"]) == (ds, s, m) and r.get("grad_norms")]
            if not logs:
                continue
            per = {}
            for g in logs:
                for e in g:
                    if e.get("ce"):
                        per.setdefault(e["epoch"], []).append(e["kernel"] / e["ce"])
            xs = sorted(per)
            ax.plot(xs, [st.mean(per[x]) for x in xs], color=c, linestyle=ls, linewidth=1.3, label=f"{lab} (n={len(logs)})")
        ax.axhline(1.0, color=INK, linewidth=0.6, linestyle=":")
        ax.set_yscale("log")
        ax.set_title(label(ds, s), fontsize=8.5, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\|\nabla L_{kernel}\|\,/\,\|\nabla L_{CE}\|$")
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    h = [Line2D([], [], color=c, linestyle=ls, linewidth=1.4, label=lab) for _, c, ls, lab in arms]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0)
    fig.text(0.99, 0.065, "dotted = equal gradient size; pqk, random target: TrashNet/MobileNetV2 only (n = 2 logged)",
             ha="right", color=MUTED, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "gradient_balance")


def fig_int8_delta():
    """INT8 - fp32 macro-F1 for every reported model, by method, under PC and Pi-XNNPACK numerics."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
    for ax, (num, title) in zip(axes, (("pc", "PC (builtin kernels)"), ("xnnpack", "Pi 4B, XNNPACK (default)"))):
        for x, m in enumerate(agg.METHODS):
            vals = []
            for j, (ds, s) in enumerate(EDGE_PAIRS):
                sc = agg.numerics_scores(ds, s, agg.exported_variant(ds, s) if m == "pqk" else m)
                d = [sc[num][k] - sc["fp32"][k] for k in sorted(sc[num])]
                vals += d
                ax.scatter([x + (j - 1.5) * 0.09 + (k - 1) * 0.025 for k in range(len(d))], d, s=9,
                           color=BLUE if m == "pqk" else GRAY, alpha=0.55, linewidths=0, zorder=3)
            ax.hlines(st.mean(vals), x - 0.3, x + 0.3, color=BLUE if m == "pqk" else INK, linewidth=1.8, zorder=4)
        ax.axhline(0, color=INK, linewidth=0.7, zorder=1)
        ax.set_xticks(range(len(agg.METHODS)))
        ax.set_xticklabels(agg.METHODS, rotation=30, ha="right", fontsize=7.5)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("INT8 $-$ fp32 macro-F1")
    fig.text(0.99, 0.005, "each dot = one model (4 pairs × 3 seeds = 12 per method); bar = mean", ha="right", va="bottom",
             color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "int8_delta")


# ----------------------------------------------------------------------------- on-device figures
def _edge():
    return agg.EDGE


def fig_latency():
    E = _edge()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    cols = {"mobilenetv2_035": BLUE, "mobilenetv3_small": ORANGE}
    for s, c in cols.items():
        pts = {}
        for r in E:
            if r["precision"] == "int8" and r["resolver"] == "auto" and r["student"] == s and r["method"] == "pqk" and r["seed"] == 0:
                pts.setdefault(r["threads"], []).append(r["invoke_ms"]["p50"])
        xs = sorted(pts)
        a1.plot(xs, [st.mean(pts[x]) for x in xs], marker="o", color=c, linewidth=1.4, markersize=4.5)
        a1.text(xs[-1] + 0.08, st.mean(pts[xs[-1]]), agg.SHORT[s], color=c, va="center", fontsize=7.5)
    a1.set_xlabel("threads")
    a1.set_ylabel("p50 latency (ms)")
    a1.set_xticks([1, 2, 3, 4])
    a1.set_xlim(0.8, 5.6)
    a1.set_ylim(0, 5.5)
    a1.set_title("(a) INT8 thread scaling", fontsize=8.5, loc="left")
    a1.grid(axis="y", color=GRID, linewidth=0.5)
    a1.set_axisbelow(True)
    groups = [(s, t) for s in agg.EDGE_STUDENTS for t in (1, 4)]
    for i, (s, t) in enumerate(groups):
        f = [r["invoke_ms"]["p50"] for r in E if r["precision"] == "fp32" and r["student"] == s and r["threads"] == t]
        q = [r["invoke_ms"]["p50"] for r in E if r["precision"] == "int8" and r["resolver"] == "auto" and r["student"] == s
             and r["threads"] == t and r["method"] == "pqk" and r["seed"] == 0]
        a2.bar(i - 0.2, st.mean(f), 0.36, color=GRAY, label="fp32" if i == 0 else None)
        a2.bar(i + 0.2, st.mean(q), 0.36, color=BLUE, label="INT8" if i == 0 else None)
        a2.text(i + 0.2, st.mean(q) + 0.1, f"{st.mean(f) / st.mean(q):.2f}×", ha="center", fontsize=7, color=INK)
    a2.set_xticks(range(len(groups)))
    a2.set_xticklabels([f"{FSHORT[s]}\n{t} thr." for s, t in groups], fontsize=7.5)
    a2.set_ylabel("p50 latency (ms)")
    a2.set_ylim(0, 7)
    a2.set_title("(b) fp32 vs INT8 (XNNPACK)", fontsize=8.5, loc="left")
    a2.legend(frameon=False, fontsize=7.5, loc="upper right")
    a2.grid(axis="y", color=GRID, linewidth=0.5)
    a2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "latency")


def fig_camera():
    """Per-frame processing time of the live camera pipeline, split into its stages (p50)."""
    cams = []
    for p in glob.glob(os.path.join(agg.REPO, "edge", "results", "rpi4", "camera", "*", "camera.json")):
        cams.append(json.load(open(p, encoding="utf-8")))
    cams.sort(key=lambda c: (c["student"], c["threads"], str(c["cv_threads"]), c["method"]))
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    comps = [("preprocess_ms", "image prep. (OpenCV)", ORANGE), ("invoke_ms", "inference (TFLite)", BLUE),
             ("postprocess_ms", "post-processing", GRAY)]
    for i, c in enumerate(cams):
        y = len(cams) - 1 - i
        left = 0.0
        for key, _, col in comps:
            w = c[key]["p50"]
            ax.barh(y, w - 0.12 if w > 0.3 else w, left=left, height=0.62, color=col)
            left += w
        ax.text(left + 0.3, y, f"{left:.1f} ms  ({c['cpu']['process_util_pct']:.0f}% CPU)", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(cams)))
    ax.set_yticklabels([f"{FSHORT[c['student']]} · {c['method']} · TFLite {c['threads']}t, OpenCV "
                        f"{'default' if 'cvdefault' in c['config'] else str(c['cv_threads']) + 't'}"
                        for c in reversed(cams)], fontsize=7)
    ax.set_xlabel("per-frame processing time, p50 (ms)")
    ax.set_xlim(0, 36)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    h = [Rectangle((0, 0), 1, 1, color=col, label=lab) for _, lab, col in comps]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "camera_pipeline")


def fig_gradvar():
    p = os.path.join(GEN, "gradient_variance.json")
    if not (os.path.exists(p) and os.path.getsize(p)):
        print("skip gradvar: run scripts/paper_diagnostics.py")
        return
    G = json.load(open(p, encoding="utf-8"))
    n = sorted(int(k) for k in G)
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.plot(n, [G[str(k)]["variance"] for k in n], marker="o", color=BLUE, linewidth=1.4, markersize=4.5)
    ax.axhline(1e-6, color=ORANGE, linestyle="--", linewidth=1)
    ax.text(n[-1], 1.6e-6, "barren-plateau flag (var < $10^{-6}$)", ha="right", color=ORANGE, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 10)
    ax.set_xticks(n)
    ax.set_xlabel("qubits (depth 4, 200 random inputs)")
    ax.set_ylabel(r"Var$[\partial\langle Z_0\rangle/\partial\theta_0]$")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "barren_plateau")


# ----------------------------------------------------------------------------- data samples
def fig_samples():
    from PIL import Image
    rng = __import__("random").Random(0)
    with open(os.path.join(agg.REPO, "data", "splits", "trashnet.json"), encoding="utf-8") as f:
        tn = json.load(f)
    with open(os.path.join(agg.REPO, "data", "splits", "rps_25.json"), encoding="utf-8") as f:
        rp = json.load(f)

    def pick(split, label_idx, k):
        c = [e["path"] for e in split["train"] if e["label"] == label_idx]
        return rng.sample(c, k)

    def load(root, rel):
        return Image.open(os.path.join(agg.REPO, "data", "raw", root, rel)).convert("RGB").resize((128, 128))

    fig = plt.figure(figsize=(6.5, 4.6))
    top = fig.add_gridspec(2, 6, hspace=0.04, wspace=0.04, top=0.88, bottom=0.40, left=0.02, right=0.98)
    bot = fig.add_gridspec(1, 6, wspace=0.04, top=0.285, bottom=0.04, left=0.02, right=0.98)
    for j, name in enumerate(tn["class_names"]):
        n = sum(1 for e in tn["train"] if e["label"] == j)
        for r, p in enumerate(pick(tn, j, 2)):
            ax = fig.add_subplot(top[r, j])
            ax.imshow(load("TrashNet", p))
            ax.axis("off")
            if r == 0:
                ax.set_title(f"{name} (n={n})", fontsize=7)
    for j, name in enumerate(rp["class_names"]):
        for r, p in enumerate(pick(rp, j, 2)):
            ax = fig.add_subplot(bot[0, 2 * j + r])
            ax.imshow(load("Rock-Paper-Scissors", p))
            ax.axis("off")
            if r == 0:
                ax.set_title(f"{name} (n=210)", fontsize=7, loc="left")
    fig.text(0.02, 0.985, "TrashNet — 6 classes, imbalanced", fontsize=8, fontweight="bold", va="top")
    fig.text(0.02, 0.375, "RPS-25 — 3 classes, 25% training subset", fontsize=8, fontweight="bold", va="top")
    fig.text(0.98, 0.003, "training images at the 128×128 model input resolution", ha="right", fontsize=6.3, color=MUTED)
    save(fig, "dataset_samples")


# ----------------------------------------------------------------------------- diagrams (drawn in code)
def _box(ax, x, y, w, h, text, fc="white", ec=MUTED, fs=7.2, lw=0.9, ls="-", tc=INK, bold=False, lsp=1.15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.06", fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4,
            fontweight="bold" if bold else "normal", linespacing=lsp)


def _arrow(ax, x1, y1, x2, y2, color=MUTED, ls="-", rad=0.0, lw=1.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2, shrinkA=0, shrinkB=0))


def fig_pipeline():
    """Training-time pipeline: what shapes the student, and what is exported."""
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)  # let the diagram fill the page width
    fs, lsp = 7.8, 1.4
    y_t, y_s, h = 6.35, 2.25, 1.45  # teacher row, student row, box height
    cy_t, cy_s = y_t + h / 2, y_s + h / 2

    # training-only region (everything inside is discarded at export)
    ax.add_patch(FancyBboxPatch((4.6, 0.95), 6.8, 7.95, boxstyle="round,pad=0.01,rounding_size=0.12", fc=PAPER, ec=GRAY, lw=0.9,
                                ls=(0, (4, 3)), zorder=1))
    ax.text(8.0, 8.5, "training only — discarded at export", ha="center", va="center", fontsize=8.2, color=MUTED, style="italic")

    def box(x, y, w, text, **kw):
        _box(ax, x, y, w, h, text, fs=fs, lsp=lsp, **kw)

    # input
    _box(ax, 0.1, 4.0, 1.7, 1.45, "image batch\n128×128\n$B\\leq128$", fs=fs, lsp=lsp)
    # teacher row (frozen)
    box(2.3, y_t, 1.9, "ResNet-50\nteacher\n(frozen, eval)", fc="#eeedea")
    box(4.75, y_t, 2.3, "fixed random\nprojection\n→ 8 angles", fc="#eeedea")
    box(7.4, y_t, 1.85, "kernel map:\nPQK circuit\nor RBF", fc=LIGHT_BLUE, ec=BLUE)
    box(9.55, y_t, 1.7, "Gram matrix\n$K_t$")
    # student row (trained)
    box(2.3, y_s, 1.9, "student CNN\n(trained)", ec=INK, lw=1.4, bold=True)
    box(4.75, y_s, 2.3, "trainable\nprojection head\n→ 8 angles")
    box(7.4, y_s, 1.85, "same kernel:\nPQK circuit\nor RBF", fc=LIGHT_BLUE, ec=BLUE)
    box(9.55, y_s, 1.7, "Gram matrix\n$K_s$")
    # kernel-alignment loss between the two Gram matrices
    _box(ax, 7.9, 4.45, 3.3, 1.1, r"$\Vert\tilde K_t-\tilde K_s\Vert_F^{2}$", ec=BLUE, fs=8.8)
    # total loss
    _box(ax, 11.75, 3.95, 1.15, 2.1, r"$L=\alpha\,\mathrm{CE}$" "\n" r"$+\,\beta\tau^{2}\mathrm{KL}$" "\n" r"$+\,\gamma L_{kernel}$",
         ec=INK, fs=7.6, lsp=1.5)

    # arrows: input, rows, Gram -> loss -> total loss
    _arrow(ax, 1.8, 5.1, 2.3, cy_t)
    _arrow(ax, 1.8, 4.4, 2.3, cy_s)
    for xa, xb, y in ((4.2, 4.75, cy_t), (7.05, 7.4, cy_t), (9.25, 9.55, cy_t), (4.2, 4.75, cy_s), (7.05, 7.4, cy_s), (9.25, 9.55, cy_s)):
        _arrow(ax, xa, y, xb, y)
    _arrow(ax, 10.4, y_t, 10.4, 5.57, color=BLUE)
    _arrow(ax, 10.4, y_s + h, 10.4, 4.43, color=BLUE)
    _arrow(ax, 11.2, 5.0, 11.75, 5.0, color=BLUE)

    # notes, each in its own free space inside the dashed region
    ax.text(6.45, 5.95, "frozen: no gradient reaches\nthe teacher or its projection", fontsize=7.3, color=MUTED, ha="center", va="center",
            linespacing=1.4)
    ax.text(6.0, 4.3, "$\\gamma$ ramped $0\\to1$\nover 10 epochs", fontsize=7.5, color=BLUE, ha="center", va="center", linespacing=1.4)
    ax.text(8.0, 1.65, "CE (hard labels) and KL to the teacher logits ($\\tau$)\nuse the student's logits", fontsize=7.3, color=MUTED,
            ha="center", va="center", linespacing=1.4)

    # export
    _arrow(ax, 3.3, y_s, 3.3, 0.78, color=INK, lw=1.5)
    _box(ax, 0.1, 0.1, 12.8, 0.65, "export: student CNN → ONNX → INT8 TFLite (per-channel); no quantum ops",
         fc=INK, ec=INK, tc="white", fs=7.7, bold=True)
    save(fig, "pipeline")


def fig_circuit():
    """The fixed feature map: depth x [RY(angle_i) on every qubit, then a linear CNOT chain], then <X>,<Y>,<Z> per qubit."""
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.2, 7.3)
    ax.axis("off")
    ys = {0: 6.0, 1: 5.0, 2: 4.0, 6: 2.3, 7: 1.3}
    xs, xe = 1.0, 7.5
    for q, y in ys.items():
        ax.plot([xs, xe], [y, y], color=MUTED, lw=0.8, zorder=1)
        ax.text(xs - 0.12, y, f"q$_{q}$", ha="right", va="center", fontsize=8)
    for xv in (2.02, 6.85):
        ax.text(xv, 3.15, r"$\vdots$", ha="center", va="center", fontsize=12, color=MUTED)
    for q, y in ys.items():
        _box(ax, 1.5, y - 0.3, 1.05, 0.6, rf"RY($\theta_{q}$)", fc=LIGHT_BLUE, ec=BLUE, fs=6.8)

    def cnot(x, c, t):
        ax.plot([x, x], [ys[c], ys[t]], color=INK, lw=1.0, zorder=2)
        ax.scatter([x], [ys[c]], s=16, color=INK, zorder=4)
        ax.scatter([x], [ys[t]], s=140, facecolor="white", edgecolor=INK, linewidths=1.0, zorder=3)
        ax.plot([x, x], [ys[t] - 0.09, ys[t] + 0.09], color=INK, lw=1.0, zorder=5)
        ax.plot([x - 0.09, x + 0.09], [ys[t], ys[t]], color=INK, lw=1.0, zorder=5)

    cnot(3.2, 0, 1)
    cnot(3.9, 1, 2)
    ax.text(4.6, 3.15, r"$\cdots$", ha="center", va="center", fontsize=12, color=MUTED)
    cnot(5.3, 6, 7)
    # repeat bracket
    ax.plot([1.4, 1.4, 5.7, 5.7], [0.75, 0.55, 0.55, 0.75], color=MUTED, lw=1)
    ax.text(3.55, 0.2, "repeated depth $=4$ times (data re-uploading, same angles)", ha="center", fontsize=7, color=MUTED)
    # measurements
    for q, y in ys.items():
        _box(ax, 6.2, y - 0.3, 1.3, 0.6, r"$\langle X\rangle\langle Y\rangle\langle Z\rangle$", fc="white", ec=MUTED, fs=6.6)
    ax.text(8.0, 6.0, "Bloch vector per qubit:\nonly single-qubit expectation\nvalues are read out", fontsize=6.8,
            va="center", ha="left")
    ax.text(8.0, 3.95, r"$\langle Y_q\rangle\equiv0$ (RY and CNOT are real),", fontsize=6.8, color=MUTED, va="center", ha="left")
    ax.text(8.0, 3.55, "so 16 of the 24 coordinates are active", fontsize=6.8, color=MUTED, va="center", ha="left")
    ax.text(8.0, 2.2, r"$K_{ij}=\exp(-\lambda\cdot\frac{1}{2}\|r(x_i)-r(x_j)\|^2)$", fontsize=7, color=BLUE, va="center", ha="left")
    ax.text(8.0, 1.6, "$r$ = stacked Bloch vectors; $\\lambda{=}1$ or median heuristic", fontsize=6.4, color=MUTED, va="center", ha="left")
    ax.text(0.1, 7.18, "8 qubits, all starting in $|0\\rangle$; angles $\\theta=\\pi\\,\\sigma(Wz+b)$ from the penultimate features $z$",
            fontsize=6.6, color=MUTED, va="center")
    ax.text(0.1, 6.78, "CNOTs form an open chain $i\\to i{+}1$ (no wrap-around)", fontsize=6.6, color=MUTED, va="center")
    save(fig, "pqk_circuit")


FIGS = {"pipeline": fig_pipeline, "circuit": fig_circuit, "samples": fig_samples, "g2": fig_g2_paired, "methods": fig_methods_seeds,
        "pqk_kd": fig_pqk_vs_kd, "variance": fig_variance, "fitted": fig_fitted, "gradients": fig_gradients,
        "int8": fig_int8_delta, "latency": fig_latency, "camera": fig_camera, "gradvar": fig_gradvar}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    for name, fn in FIGS.items():
        if a.only and name not in a.only.split(","):
            continue
        try:
            fn()
        except Exception as e:  # one broken figure must not hide the rest
            print(f"FAILED {name}: {type(e).__name__}: {e}")
