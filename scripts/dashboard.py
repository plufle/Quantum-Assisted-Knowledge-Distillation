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

if not os.path.exists(RESULTS_JSON_PATH):
    st.warning("results.json not found — run `python scripts/aggregate_results.py` first.")

st.title("QAKD — Results Dashboard")
st.caption("Quantum-Assisted Knowledge Distillation for Lightweight Edge Classification")

# ---------------------------------------------------------------- Progress
st.header("Progress against the 83-run plan")


def _count_block(df, dataset, students, methods):
    if df.empty:
        return 0
    return len(df[(df.dataset == dataset) & (df.student.isin(students)) & (df.method.isin(methods))])


ALL_METHODS = ["scratch", "kd", "rkd", "rbf_control", "pqk"]
NO_RKD = ["scratch", "kd", "rbf_control", "pqk"]

progress = pd.DataFrame([
    {"Block": "Teachers", "Planned": 2, "Done": len(teachers_df)},
    {"Block": "Main (trashnet)", "Planned": 45,
     "Done": _count_block(students_df, "trashnet", ["mobilenetv2_035", "mobilenetv3_small", "lenet5"], ALL_METHODS)},
    {"Block": "Confirmation (rps_25)", "Planned": 24,
     "Done": _count_block(students_df, "rps_25", ["mobilenetv2_035", "mobilenetv3_small"], NO_RKD)},
    {"Block": "Quantum ablation", "Planned": 12, "Done": 0},
])
progress["Remaining"] = progress["Planned"] - progress["Done"]

total_done, total_planned = int(progress["Done"].sum()), int(progress["Planned"].sum())

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Total runs completed", f"{total_done} / {total_planned}", f"{total_done / total_planned:.0%}")
    st.dataframe(progress, use_container_width=True, hide_index=True)
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
    [("trashnet", s, "macro_f1") for s in ["mobilenetv2_035", "mobilenetv3_small", "lenet5"]]
    + [("rps_25", s, "top1_accuracy") for s in ["mobilenetv2_035", "mobilenetv3_small", "lenet5"]]
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
