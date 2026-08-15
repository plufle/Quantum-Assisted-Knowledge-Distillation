# CLAUDE.md — QAKD

Quantum-Assisted Knowledge Distillation for Lightweight Edge Classification.

## The idea

A quantum kernel shapes the student's feature space **during training only**.
The exported model is a plain INT8 CNN with zero quantum ops, running on a Raspberry Pi.
We do not claim quantum speedup.

## Rules

1. No quantum ops in the exported model. Ever.
2. Every `pqk` run needs a matched `rbf_control` run next to it.
3. All methods on a pair share the same data, optimizer, and epoch count.
4. Every number is mean ± std over 3 seeds.
5. Log training cost (GPU-hours + simulator-hours). Report it.

## Datasets

| Key | Data | Classes | Role |
|-----|------|---------|------|
| `trashnet` | ~2,500 images, imbalanced (trash ≈ 140) | 6 | **Primary** — all main results |
| `rps_25` | Rock-Paper-Scissors, 25% train subset | 3 | Confirmation + fast dev loop |

- Input resolution **128×128** for both.
- TrashNet has no official split → stratified 70/15/15, seed 0, cached to
  `data/splits/trashnet.json`. Never re-split.
- **TrashNet metric is macro-F1**, not accuracy. Report accuracy as secondary.
- RPS is used at 25% of train data on purpose. At full size a pretrained model hits
  ~99% and no KD method can show a gap. Low-data is where KD should help.
- RPS's folder layout is inconsistent between splits: `train/` and `test/` are nested
  by class (`train/rock/*.png`), but `validation/` is flat with class-prefixed filenames
  (`rock1.png`, `paper-hires1.png`, ...). Loaders must handle both (see Trap #9).

## Models

**Teacher** — one architecture, trained once per dataset, then frozen.

| Key | Model | ~Params |
|-----|-------|---------|
| `resnet50_in1k` | ResNet-50, ImageNet-pretrained, fine-tuned | 25.6 M |

Pretrained weights are mandatory — 2.5k images is not enough to train a teacher from scratch.
Checkpoints: `checkpoints/teachers/resnet50_in1k__{dataset}.pt` + sidecar JSON with
macro-F1, recipe, git SHA.

**Students** — 4 architectures.

| Key | Model | ~Params | Role |
|-----|-------|---------|------|
| `lenet5` | LeNet-5 + AdaptiveAvgPool before FC | ~0.07 M | capacity floor, dev loop |
| `mobilenetv2_035` | MobileNetV2 ×0.35 | ~0.4 M | **primary edge target** |
| `mobilenetv3_small` | MobileNetV3-Small | ~1.0 M | **primary edge target** |
| `shufflenetv2_050` | ShuffleNetV2 ×0.5 | ~0.4 M | cross-family control |

Every student implements:
```python
def forward(self, x, return_feats=False):  # -> logits, or (logits, [f1..f4, penult])
```

## Methods

Build in order. Don't touch quantum code until stage 1 works.

- **Stage 0:** `scratch` (hard labels only) — the number everything is measured against
- **Stage 1:** `kd` (Hinton logit KD), `rkd` (relational KD — the direct classical ancestor)
- **Stage 2:** `rbf_control` (mandatory twin), `pqk` (ours)

INT8 quantization is **not a method** — it's an evaluation step applied to every final
checkpoint. Every model in the results table gets an fp32 and an INT8 row.

## Loss

```
L = α·CE + β·τ²·KL(teacher ‖ student) + γ·L_kernel
```

`L_kernel`: project teacher and student features to 8 angles → encode as 8-qubit states →
build batch Gram matrices with the projected quantum kernel
`K_ij = exp(−λ·Σ_q ‖ρ_q(x_i) − ρ_q(x_j)‖²)` → `‖K̃_t − K̃_s‖²_F`.

Projection layer is trained and discarded at export.

Defaults: `n_qubits=8`, `depth=4`, `λ=1.0`, `γ=1.0` (ramped over 10 epochs), batch ≥ 128.

`rbf_control` is the same pipeline with an RBF kernel at matched dimension. Tune its
bandwidth with the same budget you gave the quantum branch.

## Experiments — 95 runs

Run ID: `{dataset}__{student}__{method}__s{seed}` (teacher is always `resnet50_in1k`).
Results go to `results/{run_id}/`. Immutable.

| Block | Grid | Runs |
|-------|------|------|
| Teachers | 1 model × 2 datasets | 2 |
| **Main** | trashnet × {mnv2_035, mnv3_small, lenet5} × 5 methods × 3 seeds | 45 |
| Confirmation | rps_25 × {mnv2_035, mnv3_small} × 4 methods × 3 seeds | 24 |
| Cross-family | trashnet × shufflenetv2_050 × 4 methods × 3 seeds | 12 |
| Quantum ablation | rps_25, n_qubits {4,6,8} × kernel {fidelity, pqk} × 2 seeds | 12 |
| Edge benchmark | export + measure only, no training | 0 |

Confirmation / cross-family blocks use 4 methods (drop `rkd`).

**Minimum viable — 44 runs** if time is tight: teachers (2) + trashnet 2 students × 4
methods × 3 seeds (24) + rps_25 1 student × 4 methods × 3 seeds (12) + ablation (6).

**Cost:** ~2.5k images at 128×128 with a pretrained backbone → 10–20 min per run.
Full plan ≈ 20–25 GPU-hours. A weekend on one RTX 3060.

Metrics: macro-F1 (primary, trashnet), top-1, params, INT8 KB, MACs,
p50/p95 latency, mJ/inference, ECE.

## Go/no-go

| Gate | Criterion | If it fails |
|------|-----------|-------------|
| G1 | `kd` beats `scratch` by a clear margin on trashnet | Fix the pipeline first |
| G2 | `pqk` beats `rbf_control` by >0.5% macro-F1, 2 students, 3 seeds | Write it up as a negative result + edge benchmark |
| G3 | `pqk` INT8 beats `kd` INT8 at matched KB | Drop the edge claim from the title |

**Run G2 in week 2** on lenet5/rps_25. Costs a few hours and tells you if the project
has a result.

## Deploy

```
fp32 student → strip projection + quantum modules → purity_check (fails loudly)
→ ONNX (opset 17) → parity check → TFLite INT8 per-channel → on-device benchmark
```

Latency: 50 warmup, 500 timed, batch 1, log SoC temperature.
Energy: INA219 on the 5V rail, idle subtracted.
Targets: Raspberry Pi 4B, Jetson Orin Nano. ESP32-S3 only for `lenet5` and `mobilenetv2_035`.

## Repo layout

```
qakd/
├── setup.py, requirements.txt, .gitignore
├── configs/                       hydra: dataset/ teacher/ student/ method/
├── data/
│   ├── raw/                       TrashNet/, rock_paper_scissors/ — not committed to git
│   └── splits/                    cached stratified split JSONs — never re-generate
├── notebooks/                     EDA only, no pipeline logic
├── src/qakd/
│   ├── logger.py, exception.py, utils.py    (utils.py includes seeding)
│   ├── data/           datasets.py (loaders+transforms), splits.py
│   ├── models/         registry.py, students/, teachers/
│   ├── losses/         kd.py, rkd.py, pqk.py
│   ├── quantum/        kernels.py, classical_ctrl.py
│   ├── train/          trainer.py
│   └── deploy/         export.py, purity_check.py
├── scripts/                       standalone analysis, not importable package code
├── edge/bench/                    latency + power benchmarks — outside src/ on purpose
├── checkpoints/teachers/          resnet50_in1k__{dataset}.pt + sidecar json
└── results/                       {run_id}/ — immutable, never hand-edited
```

No hardcoded hyperparameters in `src/`. Hydra configs only.

## Traps

1. Use `lightning.qubit`, not `default.qubit` — 50× slower.
2. Batch < 64 makes the relational loss pure noise. With 2.5k images, use batch 128 and
   accumulate if memory forces smaller.
3. Kernel concentration: if mean off-diagonal `K_ij` < 0.05, the loss is vacuous. Log it.
4. TrashNet class imbalance — use weighted sampling or class-weighted CE, and always
   report macro-F1.
5. INT8 depthwise convs need per-channel quantization. Per-tensor will cost points and
   look like a method failure.
6. Teacher stays in `.eval()` with `requires_grad=False`.
7. Don't exceed 8 qubits without running the gradient-variance check (barren plateaus).
8. With ~2.5k images, students overfit fast. Fix augmentation (RandAugment + random
   resized crop) once and use it identically for every method.
9. RPS `validation/` is flat (class-prefixed filenames), not nested by class like
   `train/`/`test/` — a loader that assumes uniform structure across splits will
   silently miscount or crash on validation.

## Commands

```bash
python -m qakd.train.teacher teacher=resnet50_in1k dataset=trashnet     # once per dataset, idempotent

# below lands in Stage 2 — student/method training loop not built yet
python -m qakd.train dataset=trashnet student=mobilenetv2_035 method=pqk seed=0
python -m qakd.train dataset=trashnet student=mobilenetv2_035 method=rbf_control seed=0

python -m qakd.sweep experiment=main
python scripts/gradient_variance.py --qubits 2,4,6,8,10,12
python -m qakd.deploy.export --run <id> --format tflite_int8
python edge/bench/run.py --model <path> --device rpi4
python scripts/aggregate.py --table main --format latex
```

## Progress

Update the status column as each deliverable lands. Don't start a stage before the previous
one's deliverable exists — this mirrors the build order in **Methods**.

| # | Stage | Deliverable | Status |
|---|-------|-------------|--------|
| 0 | Scaffolding | Repo layout in place; `configs/` skeleton; data loaders + stratified split cached to `data/splits/trashnet.json`; EDA notebook runs | Done |
| 1 | Teachers | `resnet50_in1k` fine-tuned on both datasets — `checkpoints/teachers/resnet50_in1k__{trashnet,rps_25}.pt` + sidecar JSON (macro-F1, recipe, git SHA) | Done |
| 2 | Stage 0/1 methods | `scratch`, `kd`, `rkd` running end-to-end on ≥1 student × both datasets, 3 seeds | Not started |
| 3 | **G1 gate** | `kd` beats `scratch` by a clear margin on trashnet — else fix the pipeline before going further | Not started |
| 4 | Quantum infra | `quantum/kernels.py` (`lightning.qubit`, 8-qubit encoding, batch Gram matrix), `quantum/classical_ctrl.py` (RBF control), `scripts/gradient_variance.py` run for 2–12 qubits | Not started |
| 5 | Stage 2 methods | `rbf_control` + `pqk` running on lenet5/rps_25, kernel concentration logged | Not started |
| 6 | **G2 gate (week 2)** | `pqk` beats `rbf_control` by >0.5% macro-F1, 2 students, 3 seeds — else write up as a negative result + edge benchmark only | Not started |
| 7 | Full sweep | Remaining runs from the 95-run grid (or the 44-run minimum-viable set) — `results/{run_id}/`, immutable | Not started |
| 8 | Deploy pipeline | `deploy/export.py` (ONNX opset 17) + `deploy/purity_check.py` + TFLite INT8 per-channel + parity check | Not started |
| 9 | **G3 gate** | `pqk` INT8 beats `kd` INT8 at matched KB — else drop the edge claim from the title | Not started |
| 10 | Edge benchmark | `edge/bench/run.py` on Raspberry Pi 4B, Jetson Orin Nano, ESP32-S3 (lenet5/mobilenetv2_035 only) — latency, mJ/inference, SoC temp | Not started |
| 11 | Aggregation | `scripts/aggregate.py` — final results table (macro-F1, top-1, params, INT8 KB, MACs, latency, mJ, ECE) across all methods × students × datasets | Not started |

## Out of scope

Quantum ops at inference. Quantum advantage claims. >12 qubits. Federated quantum training.
