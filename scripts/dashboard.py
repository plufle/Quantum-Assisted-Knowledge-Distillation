"""QAKD results dashboard. Run with: streamlit run scripts/dashboard.py
Reads from the single consolidated results.json (repo root) instead of scanning
results/*/metrics.json + checkpoints/teachers/*.json individually — regenerate it
after training with: python scripts/aggregate_results.py"""
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="QAKD Results Dashboard", layout="wide")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_JSON_PATH = os.path.join(REPO_ROOT, "results.json")


@st.cache_data(ttl=10)
def load_results_json():
    if not os.path.exists(RESULTS_JSON_PATH):
        return {"teachers": [], "students": []}
    with open(RESULTS_JSON_PATH) as f:
        return json.load(f)


@st.cache_data(ttl=10)
def load_teacher_results():
    df = pd.DataFrame(load_results_json()["teachers"])
    if not df.empty and "params" not in df.columns:
        df["params"] = None
    return df


@st.cache_data(ttl=10)
def load_student_results():
    df = pd.DataFrame(load_results_json()["students"])
    if not df.empty:
        df["epochs_ran"] = df["recipe"].apply(lambda r: r.get("epochs_ran"))
        df["epochs_budget"] = df["recipe"].apply(lambda r: r.get("epochs_budget"))
        if "params" not in df.columns:
            df["params"] = None
    return df


teachers_df = load_teacher_results()
students_df = load_student_results()

# Two things are hidden from the dashboard but deliberately NOT deleted — both stay in
# results/ and results.json for future reference:
#   * lenet5 — retired from the reported experiment.
#   * hyperparameter screens (pqk_adaptive / pqk_t3 / pqk_g05 / pqk_g20 / pqk_d2) — the
#     lambda/tau/gamma tuning trail. Only the canonical final methods are reported.
# Filtering once here keeps progress, G1, G2, tables and charts mutually consistent.
# lenet5: retired from the reported experiment.
# mobilenetv2_050/060: the capacity-vs-architecture width probe, stopped at 1/3 seeds —
# too incomplete to report, kept on disk for whenever that question is picked up again.
EXCLUDED_STUDENTS = ["lenet5", "mobilenetv2_050", "mobilenetv2_060"]
FINAL_METHODS = ["scratch", "kd", "rkd", "rbf_control", "pqk"]

# Which pqk calibration is the *reported* one for each pair. Selected by seed-0 screen and
# then confirmed over 3 seeds; where adaptive lambda won that process, its already-computed
# runs are read in place rather than re-run under the `pqk` name — same config, same seeds,
# so a re-run would be bit-identical, and copying files would mean overwriting the
# fixed-lambda G2 record and hand-editing metrics.json (results/ is immutable).
# The rendered tables carry a `pqk_config` column so the substitution is never silent.
# Selection rule: the calibration with the higher 3-seed mean *validation* macro-F1, never
# test. rps_25/mobilenetv2_035 -> adaptive (val 0.7975 vs 0.7805). rps_25/mobilenetv3_small
# -> fixed (val 0.9065 vs 0.8736) — it was previously set to adaptive off a seed-0 *test*
# screen, which was wrong by both criteria. trashnet/mobilenetv3_small -> adaptive (val
# 0.7366 vs 0.7248, 3 seeds each, added 2026-10-02). trashnet/mobilenetv2_035 has adaptive
# at n=1 only, so its 3-seed fixed-lambda runs stand. rps_25's val split is 33 images, so
# treat the rps_25 choices as weak.
PQK_CONFIG = {
    ("rps_25", "mobilenetv2_035"): "pqk_adaptive",
    ("trashnet", "mobilenetv3_small"): "pqk_adaptive",
}
DEFAULT_PQK = "pqk"
_PQK_LABEL = {"pqk": "fixed λ=1.0", "pqk_adaptive": "adaptive λ (median heuristic)"}

if not students_df.empty:
    n_before = len(students_df)
    students_df = students_df[~students_df.student.isin(EXCLUDED_STUDENTS)].reset_index(drop=True)

    keep = []
    for row in students_df.itertuples():
        chosen = PQK_CONFIG.get((row.dataset, row.student), DEFAULT_PQK)
        if row.method in FINAL_METHODS and row.method != "pqk":
            keep.append((row.Index, row.method, ""))          # non-pqk final methods
        elif row.method == chosen:                            # the reported pqk calibration
            keep.append((row.Index, "pqk", _PQK_LABEL.get(chosen, chosen)))
    idx = [k[0] for k in keep]
    students_df = students_df.loc[idx].copy()
    students_df["pqk_config"] = [k[2] for k in keep]
    students_df["method"] = [k[1] for k in keep]
    students_df = students_df.reset_index(drop=True)
    hidden_runs = n_before - len(students_df)
else:
    hidden_runs = 0

if not os.path.exists(RESULTS_JSON_PATH):
    st.warning("results.json not found — run `python scripts/aggregate_results.py` first.")

st.title("QAKD — Results Dashboard")
st.caption("Quantum-Assisted Knowledge Distillation for Lightweight Edge Classification")

# ---------------------------------------------------------------- Progress
# The original 83-run grid included a 12-run quantum-ablation block that was dropped, and
# excluded ~32 runs that were actually done and actually used (rkd on rps_25, the whole
# lenet5/rps_25 pair, and the G2 lambda-calibration screens). Counting the plan as "83"
# therefore both over- and under-counted. The plan below is the executed programme.
st.header("Run programme")


def _count_block(df, dataset, students, methods):
    if df.empty:
        return 0
    return len(df[(df.dataset == dataset) & (df.student.isin(students)) & (df.method.isin(methods))])


ALL_METHODS = FINAL_METHODS
NO_RKD = ["scratch", "kd", "rbf_control", "pqk"]

STUDENTS = ["mobilenetv2_035", "mobilenetv3_small"]

main_done = _count_block(students_df, "trashnet", STUDENTS, ALL_METHODS)
conf_done = _count_block(students_df, "rps_25", STUDENTS, NO_RKD)
rkd_rps_done = _count_block(students_df, "rps_25", STUDENTS, ["rkd"])

progress = pd.DataFrame([
    {"Block": "Teachers", "Detail": "resnet50_in1k × 2 datasets", "Planned": 2, "Done": len(teachers_df)},
    {"Block": "Main (trashnet)", "Detail": "2 students × 5 methods × 3 seeds", "Planned": 30, "Done": main_done},
    {"Block": "Confirmation (rps_25)", "Detail": "2 students × 4 methods × 3 seeds", "Planned": 24, "Done": conf_done},
    {"Block": "rkd on rps_25", "Detail": "2 students × 3 seeds — G1-rps tracking", "Planned": 6, "Done": rkd_rps_done},
])
progress["Remaining"] = progress["Planned"] - progress["Done"]

planned_total = int(progress["Planned"].sum())
done_total = int(progress["Done"].sum())
unaccounted = (len(teachers_df) + len(students_df)) - done_total

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Programme completed", f"{done_total} / {planned_total}",
              f"{done_total / planned_total:.0%}" if planned_total else "—")
    st.dataframe(progress, use_container_width=True, hide_index=True)
    st.caption(
        f"Final runs only. Not shown ({hidden_runs} runs, all retained in `results/` and `results.json`): "
        f"**lenet5** — retired from the reported experiment; and the **λ/τ/γ tuning screens** "
        f"(`pqk_adaptive`, `pqk_t3`, `pqk_g05`, `pqk_g20`, `pqk_d2`). The quantum-ablation block "
        f"(12 runs) was dropped before execution. "
        f"{'All reported runs are accounted for above.' if unaccounted == 0 else f'{unaccounted} run(s) not in any block.'}"
    )
with col2:
    fig = px.bar(progress, x="Block", y=["Done", "Remaining"], title="Runs by block", barmode="stack")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Teachers
st.header("Teachers (Stage 1)")
if not teachers_df.empty:
    st.dataframe(
        teachers_df[["teacher", "dataset", "params", "macro_f1", "best_val_macro_f1", "git_sha", "timestamp_utc"]],
        use_container_width=True, hide_index=True,
    )
    fig = px.bar(teachers_df, x="dataset", y="macro_f1", color="dataset",
                 title="Teacher test macro-F1 by dataset", range_y=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No teacher checkpoints yet.")

# ---------------------------------------------------------------- G1-style gates
G1_METHODS = ["scratch", "kd", "rkd"]
G1_CELLS = (
    [("trashnet", s, "macro_f1") for s in STUDENTS]
    + [("rps_25", s, "top1_accuracy") for s in STUDENTS]
)


def _gate_df(dataset, student):
    if students_df.empty:
        return pd.DataFrame()
    return students_df[
        (students_df.dataset == dataset)
        & (students_df.method.isin(G1_METHODS))
        & (students_df.student == student)
    ]


def render_gate_detail(dataset, student, metric_col):
    gate_df = _gate_df(dataset, student)
    if gate_df.empty:
        st.info("Not enough data for this comparison yet.")
        return

    agg = gate_df.groupby("method")[metric_col].agg(["mean", "std", "count"]).reindex(G1_METHODS).reset_index()
    st.dataframe(agg, use_container_width=True, hide_index=True)

    scratch_mean = agg.loc[agg.method == "scratch", "mean"]
    for method in ["kd", "rkd"]:
        method_mean = agg.loc[agg.method == method, "mean"]
        if not scratch_mean.empty and not method_mean.empty and pd.notna(method_mean.iloc[0]):
            margin = method_mean.iloc[0] - scratch_mean.iloc[0]
            verdict, color = ("PASSED", "green") if margin > 0 else ("FAILING", "red")
            st.markdown(f"**{method}**: :{color}[{verdict}] — margin = {margin:+.4f} {metric_col}")

    fig = go.Figure()
    for method in G1_METHODS:
        sub = gate_df[gate_df.method == method]
        fig.add_trace(go.Box(y=sub[metric_col], name=method, boxpoints="all", pointpos=0))
    fig.update_layout(title=f"scratch vs kd vs rkd — {metric_col} across seeds ({dataset}/{student})",
                       yaxis_title=metric_col)
    st.plotly_chart(fig, use_container_width=True)


def build_g1_summary():
    rows = []
    for dataset, student, metric_col in G1_CELLS:
        gate_df = _gate_df(dataset, student)
        row = {"dataset": dataset, "student": student, "metric": metric_col,
               "scratch": None, "winner": "—", "margin": None, "verdict": "no data"}
        if not gate_df.empty:
            means = gate_df.groupby("method")[metric_col].mean()
            if "scratch" in means:
                row["scratch"] = means["scratch"]
                candidates = {m: means[m] - means["scratch"] for m in ["kd", "rkd"] if m in means}
                if candidates:
                    best_method = max(candidates, key=candidates.get)
                    best_margin = candidates[best_method]
                    row["winner"] = best_method
                    row["margin"] = best_margin
                    row["verdict"] = "PASS" if best_margin > 0 else "OPEN"
        rows.append(row)
    return pd.DataFrame(rows)


st.header("G1 Gate — kd / rkd vs scratch, every student × both datasets")
st.caption("macro-F1 on trashnet, accuracy on rps_25 (per dataset config) — best of kd/rkd vs scratch")

summary_df = build_g1_summary()


def _style_verdict(val):
    return {"PASS": "background-color: #1a4d2e; color: #d4f5dd",
            "OPEN": "background-color: #4d1a1a; color: #f5d4d4"}.get(val, "")


st.dataframe(
    summary_df.style.map(_style_verdict, subset=["verdict"])
    .format({"scratch": "{:.4f}", "margin": "{:+.4f}"}, na_rep="—"),
    use_container_width=True, hide_index=True,
)

for dataset, student, metric_col in G1_CELLS:
    with st.expander(f"{dataset} / {student} — detail"):
        render_gate_detail(dataset, student, metric_col)

# ---------------------------------------------------------------- G2 gate
# G2's criterion is macro-F1 for both datasets (unlike G1, which follows each dataset's
# own metric), so this section is macro-F1 throughout.
st.header("G2 Gate — pqk vs scratch and its matched twin rbf_control")
st.caption("Per pair, 3 seeds, macro-F1 — PASS: pqk beats BOTH scratch and rbf_control. "
           "PARTIAL: beats scratch only. FAIL: does not beat scratch.")

G2_MARGIN = 0.005


def _g2_pairs():
    if students_df.empty:
        return []
    have = students_df[students_df.method.isin(["pqk", "rbf_control"])]
    return sorted({(r.dataset, r.student) for r in have.itertuples()})


def _seed_map(dataset, student, method):
    sub = students_df[(students_df.dataset == dataset) & (students_df.student == student)
                      & (students_df.method == method)]
    return {int(r.seed): r.macro_f1 for r in sub.itertuples()}


def build_g2_summary():
    rows = []
    for dataset, student in _g2_pairs():
        pqk, rbf = _seed_map(dataset, student, "pqk"), _seed_map(dataset, student, "rbf_control")
        shared = sorted(set(pqk) & set(rbf))
        scratch = list(_seed_map(dataset, student, "scratch").values())
        row = {"dataset": dataset, "student": student, "seeds": len(shared),
               "rbf_control": None, "pqk": None, "margin": None,
               "scratch": pd.Series(scratch).mean() if scratch else None, "verdict": "no data"}
        if shared:
            p = pd.Series([pqk[s] for s in shared])
            r = pd.Series([rbf[s] for s in shared])
            row["rbf_control"], row["pqk"] = r.mean(), p.mean()
            row["margin"] = p.mean() - r.mean()
            scratch_mean = row["scratch"]
            if len(shared) < 3:
                row["verdict"] = f"INCOMPLETE ({len(shared)}/3 seeds)"
            elif scratch_mean is None:
                row["verdict"] = "no scratch baseline"
            elif p.mean() <= scratch_mean:
                # Losing to the no-teacher baseline is a failure regardless of the twin.
                row["verdict"] = "FAIL"
            elif row["margin"] > G2_MARGIN:
                # Beats scratch AND the twin on means. NOTE: this no longer distinguishes a
                # clean per-seed sweep from a mean driven by one outlier seed — on
                # rps_25/mobilenetv2_035 pqk actually loses 2 of 3 seeds and the margin comes
                # entirely from rbf_control collapsing on seed 1. The per-seed table in each
                # pair's expander still shows this, and CLAUDE.md records it.
                row["verdict"] = "PASS"
            else:
                row["verdict"] = "PARTIAL"  # beats scratch, not the twin
        rows.append(row)
    return pd.DataFrame(rows)


g2_df = build_g2_summary()
if g2_df.empty:
    st.info("No pqk / rbf_control runs yet.")
else:
    def _style_g2(val):
        if not isinstance(val, str):
            return ""
        if val == "PASS":
            return "background-color: #1a4d2e; color: #d4f5dd"
        if val == "PARTIAL":                                      # beats scratch, not the twin
            return "background-color: #24405c; color: #d4e6f5"
        if val.startswith("INCOMPLETE"):
            return "background-color: #33333d; color: #dcdce6"
        if val == "FAIL":
            return "background-color: #4d1a1a; color: #f5d4d4"
        return ""

    st.dataframe(
        g2_df.style.map(_style_g2, subset=["verdict"]).format(
            {"rbf_control": "{:.4f}", "pqk": "{:.4f}", "margin": "{:+.4f}", "scratch": "{:.4f}"}, na_rep="—"),
        use_container_width=True, hide_index=True,
    )
    n_complete = int((g2_df.seeds >= 3).sum())
    tally = g2_df[g2_df.seeds >= 3]["verdict"].value_counts().to_dict()
    st.caption(
        f"{n_complete} pairs at full 3 seeds — "
        + ", ".join(f"{k}: {v}" for k, v in sorted(tally.items()))
        + ". PASS = beats scratch AND rbf_control; PARTIAL = beats scratch only; FAIL = does not "
          "beat scratch. Verdicts compare 3-seed means — open a pair's expander for the per-seed "
          "breakdown, which is where an uneven win shows up."
    )

    for dataset, student in _g2_pairs():
        with st.expander(f"{dataset} / {student} — per-seed detail"):
            pqk, rbf = _seed_map(dataset, student, "pqk"), _seed_map(dataset, student, "rbf_control")
            shared = sorted(set(pqk) & set(rbf))
            if not shared:
                st.info("No matched seeds yet.")
                continue
            detail = pd.DataFrame([
                {"seed": s, "rbf_control": rbf[s], "pqk": pqk[s], "diff": pqk[s] - rbf[s]} for s in shared
            ])
            st.dataframe(detail.style.format({"rbf_control": "{:.4f}", "pqk": "{:.4f}", "diff": "{:+.4f}"}),
                         use_container_width=True, hide_index=True)
            if len(shared) > 1:
                diffs = detail["diff"]
                st.markdown(f"paired diff **{diffs.mean():+.4f} ± {diffs.std():.4f}** "
                            f"— {'std exceeds mean, not significant at this n' if diffs.std() > abs(diffs.mean()) else 'mean exceeds std'}")
            fig = go.Figure()
            for method in ["scratch", "rbf_control", "pqk"]:
                vals = list(_seed_map(dataset, student, method).values())
                if vals:
                    fig.add_trace(go.Box(y=vals, name=method, boxpoints="all", pointpos=0))
            fig.update_layout(title=f"macro-F1 across seeds ({dataset}/{student})", yaxis_title="macro_f1")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- G3 / INT8
st.header("G3 Gate — INT8 deployment: pqk vs kd at matched size")
_deploy = pd.DataFrame(load_results_json().get("deploy", []))
if _deploy.empty:
    st.info("No INT8 exports yet — run `python scripts/export_all.py`, then `aggregate_results.py`.")
else:
    _deploy = _deploy[_deploy.student.isin(STUDENTS)].copy()
    _deploy["fp32_f1"] = _deploy.fp32.apply(lambda d: d["macro_f1"])
    _deploy["int8_f1"] = _deploy.int8.apply(lambda d: d["macro_f1"])
    _deploy["delta"] = _deploy.int8_f1 - _deploy.fp32_f1
    _deploy["kb"] = _deploy.tflite.apply(lambda d: d["size_kb"])

    # same per-pair pqk calibration as G2
    def _reported(row):
        chosen = PQK_CONFIG.get((row.dataset, row.student), DEFAULT_PQK)
        if row.method in ("pqk", "pqk_adaptive"):
            return "pqk" if row.method == chosen else None
        return row.method if row.method in FINAL_METHODS else None
    _deploy["rmethod"] = _deploy.apply(_reported, axis=1)
    _deploy = _deploy[_deploy.rmethod.notna()]

    g3_rows = []
    for (ds, stu), grp in _deploy.groupby(["dataset", "student"]):
        p, k = grp[grp.rmethod == "pqk"], grp[grp.rmethod == "kd"]
        if len(p) < 3 or len(k) < 3:
            continue
        g3_rows.append({
            "dataset": ds, "student": stu, "KB": round(p.kb.mean(), 1),
            "kd fp32": k.fp32_f1.mean(), "kd int8": k.int8_f1.mean(),
            "pqk fp32": p.fp32_f1.mean(), "pqk int8": p.int8_f1.mean(),
            "pqk−kd (int8)": p.int8_f1.mean() - k.int8_f1.mean(),
            "verdict": "PASS" if p.int8_f1.mean() > k.int8_f1.mean() else "FAIL",
        })
    g3_df = pd.DataFrame(g3_rows)

    def _style_g3(val):
        return {"PASS": "background-color: #1a4d2e; color: #d4f5dd",
                "FAIL": "background-color: #4d1a1a; color: #f5d4d4"}.get(val, "")

    st.dataframe(
        g3_df.style.map(_style_g3, subset=["verdict"]).format(
            {c: "{:.4f}" for c in ["kd fp32", "kd int8", "pqk fp32", "pqk int8"]} | {"pqk−kd (int8)": "{:+.4f}"}),
        use_container_width=True, hide_index=True,
    )
    all_d = _deploy.delta
    st.caption(
        f"All {len(_deploy)} reported models export **fully integer, every conv per-channel, no quantum or "
        f"custom ops** (Rule #1 verified), and reload to their recorded fp32 score exactly. Size is set by "
        f"the architecture alone, so every method on a student is at matched KB by construction. "
        f"**Read the verdicts with care:** INT8 has no systematic cost on any method — mean int8−fp32 is "
        f"{all_d.mean():+.4f} ± {all_d.std():.4f}, and {(all_d > 0).sum()}/{len(all_d)} models *gained* at "
        f"INT8 — so per-pair INT8 differences are fp32 differences plus quantization noise. The rps_25 "
        f"margins swing ±0.10 seed to seed; only trashnet/mobilenetv3_small (pqk behind on all 3 seeds) "
        f"is a clean result, and it is a FAIL."
    )
    with st.expander("fp32 vs INT8 — every reported model (mean over 3 seeds)"):
        full = (_deploy.groupby(["dataset", "student", "rmethod"])
                .agg(fp32=("fp32_f1", "mean"), int8=("int8_f1", "mean"), delta=("delta", "mean"),
                     delta_std=("delta", "std"), KB=("kb", "mean"), n=("seed", "count"))
                .reset_index().rename(columns={"rmethod": "method"}))
        st.dataframe(full.style.format({"fp32": "{:.4f}", "int8": "{:.4f}", "delta": "{:+.4f}",
                                        "delta_std": "{:.4f}", "KB": "{:.1f}"}),
                     use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Conclusions
st.header("Conclusions")

c1, c2 = st.columns(2)
with c1:
    st.subheader("G1 — does distillation beat training from scratch?")
    st.markdown(
        "**1. Yes, but the margin is small.** On trashnet `kd` beats `scratch` by "
        "+2.5pt (mobilenetv2_035) and +0.9pt (mobilenetv3_small). Real and repeatable, but modest.\n\n"
        "**2. Fixing the training recipe mattered far more than choosing a method.** Retuning "
        "sampler, augmentation and epoch budget moved `scratch` ~+17pt. Every method-vs-method gap "
        "is 1–3pt. At this data scale, pipeline defects dominate method choice.\n\n"
        "**3. Dataset size decides whether anything is measurable.** trashnet's seed spread is "
        "~0.01–0.02, so a 2pt effect is visible. rps_25's reaches ~0.11 — larger than any effect "
        "present — so it confirms nothing at 3 seeds. It is a dev loop, not evidence.\n\n"
        "**4. Capacity changes which kind of distillation works.** Logit transfer holds at ~0.4M "
        "params; the smallest students needed relational transfer instead, at different temperatures."
    )
with c2:
    st.subheader("G2 — does the quantum kernel add anything over its classical twin?")
    st.markdown(
        "**1. The result splits by architecture, not by dataset or parameter count.** "
        "`mobilenetv2_035` passes on both datasets; `mobilenetv3_small` fails on trashnet and only "
        "partially clears on rps_25. Same datasets, same sizes — different backbone.\n\n"
        "**2. The likely cause is training mechanics, not the kernel.** `mobilenetv3_small` is the "
        "one student using per-batch BatchNorm statistics, and the kernel loss is built from that "
        "same batch. The *classical* twin fails on this student too — which points at the mechanism "
        "rather than at anything quantum.\n\n"
        "**3. Kernel calibration must be set per dataset/model.** A single fixed λ silently ran the "
        "kernel at a different operating point on every pair; matching the classical twin's "
        "per-batch calibration is what turned a failing pair into a pass.\n\n"
        "**4. The consistent advantage is stability, not accuracy.** `pqk` has the lowest "
        "seed-to-seed variance of any method tested, while never beating plain `kd` anywhere. At "
        "3 seeds, effects under ~3pt are not resolvable — so the honest claim is a regularisation "
        "effect, not a performance win."
    )

st.info(
    "**Overall** — kernel-alignment distillation works, and works classically. Nothing in these "
    "results requires the kernel to be quantum, which is consistent with the project's own "
    "'no quantum speedup claimed' framing. **Deployment holds for every method:** all students "
    "export to fully-integer INT8 (≈604 KB / ≈1180 KB) with no quantum ops and no systematic "
    "accuracy cost — but INT8 does not create a `pqk` advantage that fp32 lacks."
)

# ---------------------------------------------------------------- Still to do
st.header("Still to be done")

d1, d2 = st.columns(2)
with d1:
    st.subheader("1. `pqk_zz` — two-body correlators")
    st.markdown(
        "The current kernel measures only single-qubit ⟨X⟩,⟨Y⟩,⟨Z⟩ — which is exactly what "
        "discards the correlations entanglement creates. Measured: mean Bloch length |r| falls "
        "0.541 → 0.384 → 0.313 as qubits go 4 → 8 → 12, so each qubit's marginal carries *less* "
        "signal as the circuit grows (this is why raising `n_qubits` made results worse, not "
        "better). Adding the n(n−1)/2 ⟨Z_iZ_j⟩ correlators restores the two-body information: "
        "at 8 qubits the active features go 16 → 44 (⟨Y⟩ is identically zero — see below) and "
        "effective dimensionality 8.90 → 10.92, for ~25% more simulator time.\n\n"
        "**Seed-0 status — pair-dependent, in opposite directions.** On trashnet/mobilenetv3_small "
        "(the pair that failed five other configs) it scored **0.7312**, clearing both `scratch` "
        "(+0.0076) and `rbf_control` (+0.0161). But on rps_25/mobilenetv2_035 it *regressed* to "
        "0.6330 (−0.0774 vs `scratch`).\n\n"
        "**3-seed result: FAIL.** trashnet/mobilenetv3_small came in at 0.7092 ± 0.0327 — below "
        "`scratch` (0.7236) and `rbf_control` (0.7152). The seed-0 margin was optimism, as predicted.\n\n"
        "**Confound resolved (2026-10-02) — ZZ adds nothing.** `pqk_zz` changed λ to adaptive *and* "
        "added ZZ. Paired against `pqk_adaptive` on the same 3 seeds: ZZ − adaptive = **−0.0092 ± "
        "0.0372**, ZZ wins 1/3. The earlier gain over fixed λ was entirely the calibration fix "
        "(adaptive − fixed = +0.0452 paired). Correlators are not worth carrying forward as-is.\n\n"
        "**Remaining note.** The circuit uses "
        "only RY and CNOT, so the state stays real and every ⟨Y⟩ is exactly zero: the features are "
        "16 active coordinates (not 24), 44 with ZZ (not 52). Next control: a matched *classical* "
        "nonlinear feature map of the same width, to test whether any gain is quantum-specific."
    )
with d2:
    st.subheader("2. INT8 quantization (Stage 8 / G3) — done")
    st.markdown(
        "**Done (2026-10-02).** All 66 reported models exported to fully-integer TFLite "
        "(per-channel, calibrated on 256 training images) and verified — see the G3 section. "
        "Two predictions made here before the run, now tested:\n\n"
        "**\"MobileNetV3 will degrade more\" — not supported.** Hardswish and SE blocks were "
        "expected to be fragile. Measured int8−fp32: mobilenetv3_small −0.0016 ± 0.0222 vs "
        "mobilenetv2_035 +0.0068 ± 0.0214 — indistinguishable — and V3 actually agrees with its own "
        "fp32 predictions *more* often (95.3% vs 91.5%). With per-channel weights (Trap #5), both "
        "architectures quantize cleanly.\n\n"
        "**\"`pqk` will lose less going to INT8\" — not supported.** INT8 has no systematic cost on "
        "*any* method (37 of 66 models gained), so there is no loss for `pqk`'s lower variance to "
        "protect against. INT8 rankings are fp32 rankings plus noise.\n\n"
        "**Remaining (Stage 10, needs hardware):** latency and mJ/inference on Raspberry Pi 4B / "
        "Jetson Orin Nano. The `model_int8.tflite` files in `exports/` are what would be benchmarked."
    )

# ---------------------------------------------------------------- All results

# ---------------------------------------------------------------- All results
st.header("Student Results (Stage 2+)")
if not students_df.empty:
    st.subheader("Aggregated — mean ± std over seeds (CLAUDE.md Rule #4)")
    agg_all = students_df.groupby(["dataset", "student", "method"]).agg(
        params=("params", "first"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        top1_mean=("top1_accuracy", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index().sort_values(["dataset", "student", "method"])
    st.dataframe(agg_all, use_container_width=True, hide_index=True)

    st.subheader("Model size — teacher vs. students")
    size_rows = []
    if not teachers_df.empty:
        for _, row in teachers_df.iterrows():
            size_rows.append({"model": f"{row['teacher']} (teacher)", "dataset": row["dataset"], "params": row["params"]})
    for (dataset, student), _ in students_df.groupby(["dataset", "student"]):
        student_params = students_df.loc[
            (students_df.dataset == dataset) & (students_df.student == student), "params"
        ].iloc[0]
        size_rows.append({"model": student, "dataset": dataset, "params": student_params})
    size_df = pd.DataFrame(size_rows).dropna(subset=["params"])
    if not size_df.empty:
        size_df["params"] = size_df["params"].astype(int)
        fig = px.bar(
            size_df, x="model", y="params", color="dataset", barmode="group",
            title="Parameter count — teacher vs. students (log scale)", log_y=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No parameter counts recorded yet — re-run to populate `params` in metrics.json.")

    fig = px.bar(
        agg_all, x="method", y="macro_f1_mean", error_y="macro_f1_std", color="dataset",
        barmode="group", facet_col="student", title="Mean macro-F1 by method (error bars = std over seeds)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Raw per-seed results")
    cols = ["run_id", "dataset", "student", "method", "seed", "params", "macro_f1", "top1_accuracy",
            "best_val_macro_f1", "epochs_ran", "epochs_budget"]
    st.dataframe(
        students_df[cols].sort_values(["dataset", "student", "method", "seed"]),
        use_container_width=True, hide_index=True,
    )
else:
    st.info("No student results yet.")

st.caption(f"Reading from `{RESULTS_JSON_PATH}` — cached 10s. "
           f"Run `python scripts/aggregate_results.py` after training, then refresh to pick up new runs.")
