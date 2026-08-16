# QAKD — Quantum-Assisted Knowledge Distillation

Lightweight edge classification research project. A quantum kernel shapes a student
CNN's feature space **during training only** — the exported model is a plain INT8 CNN
with zero quantum ops, deployed to a Raspberry Pi. No quantum speedup is claimed.

This file is the practical "how do I run it" guide. Full project rules, dataset/model/
method specs, hyperparameter traps, and the authoritative stage-by-stage status live in
[CLAUDE.md](CLAUDE.md) — read that first if you're unsure *why* something is built the
way it is.

## Setup

```bash
pip install -r requirements.txt
pip install -e .              # installs the `qakd` package (src/ layout), editable
```

Data: `data/raw/TrashNet/` and `data/raw/Rock-Paper-Scissors/` must exist locally (not
committed to git — see `.gitignore`). Cached stratified splits already live in
`data/splits/*.json` and must **never** be regenerated once created — the loaders read
them automatically if present.

## Running the pipeline

### 1. Train teachers (once per dataset, idempotent)

```bash
python -m qakd.train.teacher dataset=trashnet
python -m qakd.train.teacher dataset=rps_25
```

Produces `checkpoints/teachers/resnet50_in1k__{dataset}.pt` + a sidecar JSON with test
macro-F1, the training recipe, and the git SHA. Skipped automatically (loads the
existing sidecar) if the checkpoint already exists.

### 2. Train a student

```bash
python -m qakd.train dataset=<trashnet|rps_25> student=<lenet5|mobilenetv2_035|mobilenetv3_small|shufflenetv2_050> method=<scratch|kd|rkd> seed=<0|1|2>
```

`kd`/`rkd` require the matching teacher checkpoint from step 1. Writes
`results/{dataset}__{student}__{method}__s{seed}/metrics.json`. Also idempotent —
`results/` is immutable, an existing run is loaded and returned, never overwritten.

`method=pqk` and `method=rbf_control` aren't implemented yet — they land in Stage 5
(quantum infra comes first, per CLAUDE.md's build order).

Any config field can be overridden on the command line, e.g.
`python -m qakd.train dataset=trashnet student=mobilenetv2_035 method=kd seed=0 student.epochs=10`.

### 3. Dashboard

```bash
streamlit run scripts/dashboard.py
```

Opens at http://localhost:8501. Reads live from `results/*/metrics.json` and
`checkpoints/teachers/*.json` (10s cache) — refresh the page to pick up new runs.
Shows per-run and aggregated (mean±std) results, plus the G1/G2/G3 go/no-go gate
verdicts computed directly from the data.

### 4. Reset + re-run the full student sweep

```powershell
.\scripts\reset_and_train_students.ps1
```

**Destructive** — clears `logs/` and `results/` (student run outputs only; teacher
checkpoints and cached data splits are untouched), then re-runs the full student sweep
end to end. Read the script header before running it. Takes a while on CPU — each
(dataset, method, seed) combination is a full training run (tens of minutes apiece for
`trashnet`/`kd`/`rkd`, given the extra teacher forward pass per batch).

## Current status

See the **Progress** table in [CLAUDE.md](CLAUDE.md) for the authoritative, up-to-date
stage-by-stage status. As of this writing: both teachers trained (Stage 1 done), the
G1 gate has passed on trashnet/mobilenetv2_035 (`kd` 0.557±0.024 vs `scratch`
0.527±0.014 macro-F1), and the Stage 2 student sweep is in progress.

## Repo layout

Package code lives under `src/qakd/` (Hydra configs in `configs/`, never hardcoded
hyperparameters). See CLAUDE.md's **Repo layout** section for the full annotated tree.
