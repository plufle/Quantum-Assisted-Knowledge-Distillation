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

# ---------------------------------------------------------------- Follow-up controls
st.header("Follow-up controls — does anything here need the quantum circuit?")
st.caption(
    "Exploratory. These four pairs' test scores already informed earlier design choices, so "
    "**validation macro-F1 decides** and test is descriptive only. trashnet validation has 379 "
    "images; rps_25 has 33, so rps_25 rows are weak. Every arm is paired seed-for-seed (same "
    "student init and data order — verified). Fills in as runs land."
)
_raw = pd.DataFrame(load_results_json()["students"])
if not _raw.empty:
    _raw = _raw[_raw.student.isin(STUDENTS)]


def _seeds(ds, stu, method, key="best_val_macro_f1"):
    sub = _raw[(_raw.dataset == ds) & (_raw.student == stu) & (_raw.method == method)]
    return {int(r["seed"]): float(r[key]) for _, r in sub.iterrows()}


def _cell(d):
    return f"{pd.Series(list(d.values())).mean():.4f} (n={len(d)})" if d else "—"


def _paired(a, b):
    """mean(a - b) over shared seeds, with a's win count."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return "—"
    diffs = [a[s] - b[s] for s in shared]
    return f"{pd.Series(diffs).mean():+.4f} ({sum(d > 0 for d in diffs)}/{len(shared)})"


if _raw.empty:
    st.info("No results yet.")
else:
    st.subheader("1. Matched classical controls (random teacher projection)")
    st.caption(
        "**no-CNOT** = the PQK circuit with its CNOTs deleted — exactly [sin 4θ, cos 4θ], so it differs "
        "from pqk *only* by entanglement. **RFF-16** = random Fourier features drawn from the circuit's own "
        "frequency spectrum (integers −4…4), 16 wide, untuned — a generic classical map of the same function "
        "class. **rbf** = plain RBF on the 8 angles (`rbf_control`). Decision rule, fixed before results: "
        "'pqk beats classical maps in general' needs pqk ≥ both controls on validation on both trashnet pairs. "
        "Columns 'pqk − X' show mean paired difference and pqk's seed wins."
    )
    rows = []
    for ds in ["trashnet", "rps_25"]:
        for stu in STUDENTS:
            q = _seeds(ds, stu, "pqk_adaptive")
            n, r, rb = _seeds(ds, stu, "noent_control"), _seeds(ds, stu, "rff_control"), _seeds(ds, stu, "rbf_control")
            rows.append({"dataset": ds, "student": stu, "val n": 379 if ds == "trashnet" else 33,
                         "pqk": _cell(q), "no-CNOT": _cell(n), "RFF-16": _cell(r), "rbf": _cell(rb),
                         "pqk − no-CNOT": _paired(q, n), "pqk − RFF-16": _paired(q, r)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("2. Train-fitted teacher projection (LDA+PCA, frozen) vs fixed random")
    st.caption(
        "The fixed random 2048→8 teacher projection discards about half the teacher's class structure; an "
        "LDA+PCA projection fitted on the training split only (selected on validation) restores it. "
        "'fitted − random' = mean paired difference (fitted's seed wins)."
    )
    rows = []
    for ds in ["trashnet", "rps_25"]:
        for stu in STUDENTS:
            pa, pf = _seeds(ds, stu, "pqk_adaptive"), _seeds(ds, stu, "pqk_fit")
            ra, rf = _seeds(ds, stu, "rbf_control"), _seeds(ds, stu, "rbf_fit")
            rows.append({"dataset": ds, "student": stu,
                         "pqk random": _cell(pa), "pqk fitted": _cell(pf), "pqk: fitted − random": _paired(pf, pa),
                         "rbf random": _cell(ra), "rbf fitted": _cell(rf), "rbf: fitted − random": _paired(rf, ra)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Gradient balance — kernel term vs CE over training"):
        st.caption(
            "Norm of each loss term's gradient w.r.t. the student's parameters, first batch of every epoch, "
            "mean over seeds. With the random target the kernel term fades to ~0.1× CE; with the fitted target "
            "it can stay several times larger than CE and crowd out the label signal."
        )
        pair = st.selectbox("pair", [f"{d} / {s}" for d in ["trashnet", "rps_25"] for s in STUDENTS], key="gn_pair")
        ds_sel, stu_sel = [p.strip() for p in pair.split("/")]
        fig = go.Figure()
        for method in ["pqk_adaptive", "noent_control", "rff_control", "pqk_fit", "rbf_fit"]:
            sub = _raw[(_raw.dataset == ds_sel) & (_raw.student == stu_sel) & (_raw.method == method)]
            logs = [g for g in sub.get("grad_norms", pd.Series(dtype=object)) if isinstance(g, list) and g]
            if not logs:
                continue
            per_epoch = {}
            for g in logs:
                for e in g:
                    if e.get("ce"):
                        per_epoch.setdefault(e["epoch"], []).append(e["kernel"] / e["ce"])
            xs = sorted(per_epoch)
            fig.add_trace(go.Scatter(x=xs, y=[sum(per_epoch[x]) / len(per_epoch[x]) for x in xs],
                                     mode="lines", name=f"{method} (n={len(logs)})"))
        fig.add_hline(y=1.0, line_dash="dot", annotation_text="kernel = CE")
        fig.update_layout(xaxis_title="epoch", yaxis_title="|∇ kernel| / |∇ CE|", yaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Teacher-kernel class separation (validation CKTA, no training)"):
        st.caption(
            "Centered kernel-target alignment between the *teacher* Gram matrix the student must match and the "
            "label kernel; 1 = encodes the class partition perfectly. Projections fit on train, scored on "
            "validation (`scripts/teacher_kernel_diagnostics.py`). Caveat: it mispredicted Block A (no-CNOT "
            "target > pqk's, yet pqk won on trashnet), so it describes the target, not student accuracy."
        )
        st.dataframe(pd.DataFrame([
            {"target kernel": "raw teacher, 2048-d RBF", "trashnet random": 0.720, "trashnet fitted": None,
             "rps_25 random": 0.864, "rps_25 fitted": None},
            {"target kernel": "angles → RBF (rbf)", "trashnet random": 0.363, "trashnet fitted": 0.825,
             "rps_25 random": 0.685, "rps_25 fitted": 0.861},
            {"target kernel": "PQK 16 (RY+CNOT)", "trashnet random": 0.245, "trashnet fitted": 0.752,
             "rps_25 random": 0.539, "rps_25 fitted": 0.797},
            {"target kernel": "no-CNOT 16", "trashnet random": 0.323, "trashnet fitted": 0.676,
             "rps_25 random": 0.642, "rps_25 fitted": 0.636},
            {"target kernel": "RFF 16 (spectrum-matched)", "trashnet random": 0.125, "trashnet fitted": None,
             "rps_25 random": 0.362, "rps_25 fitted": None},
        ]).style.format(precision=3, na_rep="—"), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Conclusions
st.header("Conclusions")
st.caption(
    "**Basis:** 152 completed student runs, 2 teachers and 66 INT8 exports; every reported cell has 3 seeds. "
    "**Status: exploratory.** The four reported pairs' test splits shaped earlier design choices, so validation "
    "decides (trashnet n=379, rps_25 n=33 — rps_25 is weak) and test is descriptive. No on-device latency or "
    "energy has been measured; INT8 accuracy comes from the TFLite interpreter on a CPU."
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("1. Dataset, architecture and method interact")
    st.markdown(
        "**The dataset decides which kind of transfer pays.** On trashnet (6 classes, 1,768 training images) "
        "logit KD is first on test for both students (0.708, 0.732). On rps_25 (3 classes, 630 images) KD is never "
        "first; relational or kernel transfer leads — `rkd` on mobilenetv2_035 (val 0.862), `rbf_control` on "
        "mobilenetv3_small (val 0.927, test 0.888). A plausible reason, untested: with three classes the "
        "teacher's soft labels add little beyond the label, so sample-to-sample geometry carries more of what "
        "the teacher knows.\n\n"
        "**The architecture decides which kernel wins.** On validation `pqk` beats `rbf_control` on "
        "mobilenetv2_035 on *both* datasets (0.705 vs 0.661; 0.797 vs 0.733), and `rbf_control` beats `pqk` on "
        "mobilenetv3_small on *both* (0.750 vs 0.737; 0.927 vs 0.907). This is **not** the BatchNorm regime, as "
        "earlier hypothesised: on rps_25 both students use per-batch BN statistics, yet the split persists. "
        "Whether it is feature width (1,280 vs 576) or the Hardswish/SE blocks is untested.\n\n"
        "**No method is best everywhere.** The four pairs have three different validation leaders (`kd`, "
        "`rbf_control` twice, `rkd`). mobilenetv3_small gains least from any teacher — `scratch` is second on "
        "validation on both datasets. And the training recipe mattered more than the method: retuning sampler, "
        "augmentation and epochs moved `scratch` ~+17pt on trashnet, more than any method-to-method gap."
    )
with c2:
    st.subheader("2. Average versus consistency")
    st.markdown(
        "**The observation that `pqk` is more consistent holds on trashnet validation — not elsewhere.** With "
        "adaptive λ it is the steadiest method on trashnet validation for both students (seed std 0.0058 and "
        "0.0028, vs 0.0175 and 0.0122 for the steadiest classical method). On rps_25 it is among the *least* "
        "consistent (5th–6th of 6), and on test `rkd` is steadiest on 3 of 4 pairs.\n\n"
        "**Why it plausibly holds on trashnet:** the kernel term pulls hardest in the first epochs (0.5–1.2× the "
        "CE gradient) — exactly when seeds diverge — then fades to ~0.1×. Adaptive λ pins the kernel's operating "
        "point every batch, which is why it is steadier than fixed λ (std 0.0141 / 0.0118).\n\n"
        "**Why not on rps_25:** 5 batches per epoch make each batch Gram matrix a noisy estimate, and checkpoint "
        "selection on a 33-image validation loss adds noise that swamps any method-level stability.\n\n"
        "**Consistency is not accuracy:** `pqk` leads no pair on mean validation score — its best placing is "
        "second (trashnet/mobilenetv2_035). **Caveat:** with 3 seeds a variance ratio must exceed ~19 for "
        "p<0.05; of 16 comparisons only one comes close (18.3×, p≈0.05). This is a tendency, not a demonstrated "
        "property."
    )

c3, c4 = st.columns(2)
with c3:
    st.subheader("3. Quantization (fp32 → INT8)")
    st.markdown(
        "**Every reported model quantizes cleanly:** 66/66 fully integer at ≈0.6 MB (mobilenetv2_035) / ≈1.2 MB "
        "(mobilenetv3_small), with no systematic accuracy cost (mean change +0.0026 ± 0.0220; 37/66 gained).\n\n"
        "**The observation that `pqk` rises while `kd` falls holds clearly on trashnet/mobilenetv2_035** — `pqk` "
        "up on 3/3 seeds (+0.0129), `kd` down on 2/3 (−0.0057) — enough to put `pqk` INT8 above `kd` INT8 there "
        "(0.707 vs 0.703). Pooled over all pairs, `pqk` has the most favourable change of the five methods (up on "
        "9/12 seeds, +0.0080; `kd` 5/12, −0.0031). **It does not hold** on trashnet/mobilenetv3_small (both fall), "
        "and the paired `pqk`−`kd` difference is not significant (p=0.22–0.38). A plausible mechanism — tighter "
        "feature geometry leaving fewer borderline predictions for 8-bit rounding to flip — is untested."
    )
with c4:
    st.subheader("4. What the quantum part does — and doesn't")
    st.markdown(
        "**The entanglement contributes, relative to its own circuit.** On trashnet `pqk` beats the identical "
        "circuit with its CNOTs removed on 6/6 validation seeds (+0.049, +0.007); on rps_25 the direction "
        "reverses (33-image validation).\n\n"
        "**Not shown:** that `pqk` beats classical feature maps in general — the generic RFF control was not "
        "run, and plain RBF on the same angles still splits 2–2 with `pqk`.\n\n"
        "**Changes that did not help:** ⟨ZᵢZⱼ⟩ correlators (−0.009 vs adaptive λ, 1/3 seeds); a train-fitted "
        "teacher projection (better on 1 of 8 arm-pairs) — on training batches it collapses to the label kernel "
        "(CKTA 0.966), duplicating CE instead of transferring relational structure. The circuit's ⟨Y⟩ readouts "
        "are identically zero, so `pqk` reads 16 active features, not 24."
    )

with st.container(border=True):
    st.markdown(
        "#### Conditional conclusion\n"
        "- **Highest mean accuracy on trashnet:** logit KD, best on mobilenetv3_small (test 0.732 fp32 / 0.724 INT8).\n"
        "- **A ≈0.6 MB INT8 model that behaves the same run to run on trashnet:** **mobilenetv2_035 + `pqk`** is the "
        "most promising PQK combination — second on validation behind KD and ahead of `scratch` and `rbf_control` "
        "(adaptive λ, 0.705), the steadiest method on validation (adaptive λ), and level with KD after INT8 in the "
        "exported fixed-λ variant (0.707 vs 0.703). The adaptive variant has not been exported to INT8.\n"
        "- **Scarce data, few classes (rps_25):** relational or kernel transfer — `rkd` on mobilenetv2_035, "
        "`rbf_control` on mobilenetv3_small — rather than logit KD; `pqk` is not distinguished there.\n"
        "- **What the evidence supports about PQK:** its circuit contributes beyond an entanglement-free twin on "
        "trashnet; it is the steadiest method on trashnet validation; it has the most favourable INT8 behaviour. "
        "**It does not support** PQK beating KD on mean accuracy anywhere, or consistently beating a classical "
        "RBF kernel.\n"
        "- **Still uncertain:** whether `pqk` beats a generic classical map of the same function class (RFF control "
        "not run); whether the consistency and INT8 tendencies survive more seeds (all below significance at n=3); "
        "what drives the architecture split; on-device latency and energy (not measured). Every result here is "
        "exploratory — a confirmatory claim needs a fresh benchmark, a frozen protocol and ≥5 seeds, e.g. "
        "mobilenetv2_035 + `pqk` (adaptive λ) vs KD vs RBF."
    )

# ---------------------------------------------------------------- Still to do
st.header("Still to be done")
d1, d2 = st.columns(2)
with d1:
    st.subheader("In progress")
    st.markdown(
        "**Fitted-projection experiment (Block B)** and the **RFF-16 generic control** are running, "
        "sequentially — results fill the *Follow-up controls* tables above as they land.\n\n"
        "**Settled since the last review:** `pqk_zz` — two-body ⟨ZᵢZⱼ⟩ correlators add nothing once the "
        "λ-calibration confound is removed (paired ZZ − adaptive = −0.0092 ± 0.0372, 1/3 seeds); its earlier "
        "gain was the calibration fix. INT8 deployment is done (G3 section). The circuit is RY + CNOT only, "
        "so every ⟨Y⟩ is exactly zero — pqk reads 16 active coordinates, not 24."
    )
with d2:
    st.subheader("Remaining")
    st.markdown(
        "**Stage 10 — on-device benchmark** (needs a Raspberry Pi 4B / Jetson Orin Nano): latency and "
        "mJ/inference for the `exports/*/model_int8.tflite` files.\n\n"
        "**Optional — γ rebalancing for fitted projections:** the fitted target's gradient can stay several "
        "times larger than CE; rescaling γ (selected on validation) would separate 'a fitted target helps' "
        "from 'γ = 1 is too strong for it'.\n\n"
        "**Stage 11 — final aggregation table.**"
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
