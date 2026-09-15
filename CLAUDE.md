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

## Experiments — 83 runs

Run ID: `{dataset}__{student}__{method}__s{seed}` (teacher is always `resnet50_in1k`).
Results go to `results/{run_id}/`. Immutable.

| Block | Grid | Runs |
|-------|------|------|
| Teachers | 1 model × 2 datasets | 2 |
| **Main** | trashnet × {mnv2_035, mnv3_small, lenet5} × 5 methods × 3 seeds | 45 |
| Confirmation | rps_25 × {mnv2_035, mnv3_small} × 4 methods × 3 seeds | 24 |
| ~~Quantum ablation~~ | ~~rps_25, n_qubits {4,6,8} × kernel {fidelity, pqk} × 2 seeds~~ | ~~12~~ **DROPPED** (2026-09-14, user decision) — would have needed a `kernel: fidelity` branch that was never built. Grid is now **71 runs** |
| Edge benchmark | export + measure only, no training | 0 |

Confirmation block uses 4 methods (drop `rkd`).

**Minimum viable — 44 runs** if time is tight: teachers (2) + trashnet 2 students × 4
methods × 3 seeds (24) + rps_25 1 student × 4 methods × 3 seeds (12) + ablation (6).

**Cost:** ~2.5k images at 128×128 with a pretrained backbone → 10–20 min per run.
Full plan ≈ 20–25 GPU-hours. A weekend on one RTX 3060.

**Actuals (rule #5, `python scripts/training_cost.py` — recovers per-run wall-clock from
log timestamps, so it covers every run retroactively):** as of 2026-09-05, kept results
cost **27.1 GPU-hours + 2.8 simulator-hours**, with another **19.2 hours discarded** to
re-runs after pipeline fixes (Trap #13 the expensive one) — 49.1 hours total. The 20–25
GPU-hour estimate above was optimistic by roughly 2×, and that is *before* the sweep is
complete; budget for re-runs, not just the grid. Per-epoch, comparing the same seed under
the same load, `pqk` costs ~19.4 s/epoch vs `rbf_control`'s ~15.4 — the quantum simulator
adds only ~26% at 8 qubits/depth 4. The teacher forward pass, not the circuit, is the
bottleneck (`scratch` runs the same pair at 2.5–7.5 s/epoch).

Metrics: macro-F1 (primary, trashnet), top-1, params, INT8 KB, MACs,
p50/p95 latency, mJ/inference, ECE.

## Go/no-go

| Gate | Criterion | If it fails |
|------|-----------|-------------|
| G1 | `kd` beats `scratch` by a clear margin on trashnet (`rkd` tracked alongside as the Stage 1 classical-ancestor baseline, not a hard blocker) | Fix the pipeline first |
| | **PASSED on kd** (mobilenetv2_035, 3 seeds): `scratch` 0.6838±0.0141, `kd` 0.7085±0.0131 macro-F1 — +2.5pt margin. 2026-08-22 retune: the original shuffle+class-weighted-CE+strong-RandAugment recipe underfit this tiny (~1750-image), from-scratch student (`scratch` was only 0.5269±0.0136). Screened on seed 0 — switching imbalance handling to a weighted sampler (dropping the now-redundant CE class weights), softening RandomResizedCrop/RandAugment, and extending the epoch budget 40→60 took `scratch` seed0 alone from 0.4892 to 0.6942. Applied to every method via `configs/dataset/trashnet.yaml` + `configs/student/mobilenetv2_035.yaml` (CLAUDE.md rule #3). The τ=3.0 kd retune still holds on top of this. | |
| | **rkd: negative result** (mobilenetv2_035, 3 seeds): `rkd` 0.6716±0.0037 macro-F1 — *below* `scratch` (0.6838±0.0141) by ~1.2pt, despite retuning. The paper's own distance/angle weights (25/50) drowned the CE signal in this regime (0.5851); screened distance/angle∈{25/50, 5/10, 2/4} and ce_weight∈{1,2} at 5/10 on seed 0 (see `configs/method/rkd.yaml`) — 5/10 was the best candidate found (seed0 0.6758) but the confirmed 3-seed mean still falls short. `kd` remains the passing Stage 1 method; `rkd` is documented here as-is rather than tuned further, per CLAUDE.md's own screen-then-stop practice (see the kd τ=4 note above for precedent). | |
| G1-rps | Confirmation echo of G1 on rps_25: `kd` beats `scratch` by a clear margin (metric: accuracy, per `configs/dataset/rps_25.yaml`); `rkd` tracked alongside, not a hard blocker | Note the discrepancy — trashnet is still primary |
| | **INCONCLUSIVE** (mobilenetv2_035, 3 seeds): `scratch` 0.7267±0.0300, `kd` 0.7186±0.0867, `rkd` 0.7446±0.0478 top1_accuracy. Neither clears scratch by a clean margin — variance across seeds is huge (kd seed2 dropped to 0.6263, *below* scratch's own seed2 of 0.7608) and swamps any average edge; rkd's nominal +1.8pt sits inside overlapping std. 2026-08-23: fixed a real bug first (Trap #11) — rps_25's 33-image validation split made `val_f1` too noisy for checkpoint selection, so the original run had kd/rkd scoring *below* scratch (0.586/0.640 vs 0.707 seed0 acc) purely from picking a badly-generalizing epoch; switching selection to `val_loss` for `small_batch_regime` datasets fixed the ordering on seed 0 (kd 0.71, rkd 0.76, both ≥ scratch's 0.72). Also found kd's trashnet-tuned τ=3 ties scratch on rps_25's 3-class problem (0.7097) — screened τ∈{3,2,1}, τ=1 was the clear seed-0 winner (0.7984); see `configs/method/kd_rps25.yaml`. Despite both fixes, the full 3-seed confirmation is too noisy to call — rps_25's 630-image train / 33-image val split is simply high-variance at this student/method combination. Per CLAUDE.md's own screen-then-stop practice, not tuned further; trashnet remains the dataset where G1 passes cleanly (rps_25 is "confirmation + fast dev loop," not primary). 2026-08-23 addendum: tried `ce_label_smoothing=0.1` dataset-wide to fix kd's seed2 crash — reverted, it was a net negative (dropped scratch to 0.6470±0.0663 and rkd to 0.6434±0.0567; kd's own numbers barely moved). See `configs/dataset/rps_25.yaml` for detail. | |
| | **mobilenetv3_small, trashnet (3 seeds)**: `scratch` 0.7236±0.0156, `kd` 0.7324±0.0201 (+0.9pt, PASSED), `rkd` 0.7155±0.0030 (-0.8pt). No BN fix needed — trains cleanly out of the box. | |
| | **lenet5, trashnet (3 seeds)**: needed its own fix (lenet5-only — doesn't touch other students or `trashnet.yaml`) — epochs 40→60/patience 8→12 in `configs/student/lenet5.yaml` (2026-08-29), since `rkd`'s worst seed was still improving when it hit the old epoch cap. `kd` (τ=3) never recovered (underperforms `scratch` at both 40 and 60 epochs — τ=1/τ=2 retunes tried in `kd_lenet5.yaml`, best was τ=2 at 0.5094, still short). `rkd` needed lighter weights too (`configs/method/rkd_lenet5.yaml`, distance=1/angle=2 vs the default 5/10 — the default was a genuine training failure here, not BN-related, LeNet5's ~35K params can't absorb the default relational loss weight). Confirmed (3 seeds): `scratch` 0.5286±0.0209, `rkd` 0.5777±0.0156 macro-F1 — +4.9pt margin, tight non-overlapping variance (PASSED via rkd). `kd` (τ=3, canonical) stayed at 0.4022, well below scratch even at 60 epochs. **Corrected 2026-09-14** — that 0.4022 was a *single seed*, and completing seeds 1-2 for the grid shows it was the worst of the three: `kd` is **0.4838±0.0743** (0.4022/0.5477/0.5015). Still below `scratch` (0.5286±0.0209) so the verdict is unchanged — `rkd` remains the passing method here — but the gap is ~4.5pt, not the ~13pt a single seed implied, and `kd`'s seed1 (0.5477) actually exceeds `scratch`'s best seed (0.5441). A reminder that screen-then-stop conclusions drawn from seed 0 alone can be right in direction and badly wrong in magnitude (rule #4). | |
| | **mobilenetv3_small, rps_25 (3 seeds)**: `scratch` 0.8157±0.0527, `kd` 0.8432±0.0752 (+1.5pt), `rkd` 0.8593±0.0412 (+3.1pt, cleaner margin — PASSED). Clean win for both methods — this student has plenty of capacity for the small-data regime. | |
| | **lenet5, rps_25 (3 seeds) — INCONCLUSIVE**: `scratch` 0.6764±0.1084 (highly seed-variable: 0.5564/0.7054/0.7673), best `kd` found (`kd_lenet5_rps25.yaml`, τ=0.5) 0.6523±0.0164 — tighter variance but still short. Tried 5 kd variants (τ∈{3,2,1,0.5}, alpha/beta 0.3/0.7) and `rkd` at 2 weight scales (default 5/10: 0.3297; lenet5's lighter 1/2, which fixed trashnet: 0.1889, *worse* — the trashnet fix doesn't transfer). Not resolved; reported honestly per CLAUDE.md's screen-then-stop practice rather than tuned further. | |
| G2 | **Per-pair, 3 seeds, macro-F1 (revised 2026-09-14):** `pqk` > **both** `scratch` and `rbf_control` = **PASS**; `pqk` > `scratch` only = **PARTIAL**; `pqk` ≤ `scratch` = **FAIL**. (Supersedes the original ">0.5pt over `rbf_control`, 2 students" wording, which was satisfiable by picking 2 favourable pairs — see the reversal note below.) | Write it up as a negative result + edge benchmark |
| | **PASSED (2026-09-13)** — 2 students × 3 seeds, both clearing the +0.5pt bar on the post-Trap-#13 pipeline. **trashnet/mobilenetv2_035**: `pqk` 0.6938±0.0056 vs `rbf_control` 0.6708±0.0103 → **+2.30pt**, `pqk` wins **3/3** seeds, paired +0.0230±0.0158 (mean exceeds std — the clean one). **rps_25/lenet5**: `pqk` 0.6375±0.0699 vs `rbf_control` 0.5377±0.2020 → **+9.98pt**, but wins only 2/3 and paired +0.0998±0.1451 has std > mean, so that pair passes on means while remaining individually noisy. Two caveats to carry into the write-up, neither of which blocks the gate: (1) `pqk` beats `scratch` on trashnet (+1.0pt) but still **trails `kd`** (0.7085) there by 1.5pt — the gate is "beat your matched twin", not "beat KD", and `pqk` has not beaten `kd` anywhere; (2) the kernel methods run a **single un-tuned τ=0.5** for their CE+KL term everywhere (inherited from the lenet5/rps_25 screen), while `kd` gets a per-pair tuned τ (3 on trashnet, 1 on rps_25/mnv2_035). Kept uniform deliberately — rule #2 needs the twins matched, and re-tuning τ per pair would invalidate the completed runs — but it means `pqk` clears `scratch` on trashnet *despite* a temperature tuned for a different dataset, and a τ sweep is the obvious next lever if a `kd`-beating result is wanted. The consistent cross-pair signal remains **variance**: `pqk` has the lowest seed-std of any method on both pairs (0.0056 and 0.0699). | |
| | ⚠️ **G2 PASS IS QUALIFIED — a third student contradicts it (2026-09-14)**. `trashnet/mobilenetv3_small`, 3 seeds: `pqk` **0.6733±0.0235** vs `rbf_control` **0.7152±0.0368** → `pqk` is **4.19pt WORSE** than its twin and wins only **1/3** seeds (s0 +0.0045, s1 −0.0927, s2 −0.0375). Both also trail `scratch` (0.7236±0.0156) and `kd` (0.7324±0.0201) on this student. The gate as *literally specified* ("2 students, 3 seeds") still passes on mnv2_035 + lenet5 — but scoring 2 wins out of 3 students tested and reporting only the 2 would be cherry-picking, so the honest headline is: **`pqk` beats its matched twin on 2 of 3 student/dataset pairs, loses clearly on the third, and beats `kd` on none.** Note the failing student is the one carrying `force_batch_stats_only` (Trap #10) for its Hardswish/SE-block BN instability — a plausible interaction between per-batch BN statistics and a batch-Gram-based loss (both are computed over the same batch, so unstable batch statistics feed straight into the kernel), but that is an untested hypothesis, not a demonstrated cause. Do not quietly drop this pair from the write-up; if anything it is the most informative one, because it says the effect is architecture-dependent rather than general. | |
| | ⚠️ **The two 2026-09-04 rows below are SUPERSEDED** — every number in them was produced by the pre-Trap-#13 pipeline (trainable teacher-side projection) and is not comparable to anything after 2026-09-05. Kept for the record of what was built, not for their numbers. Raw results archived in `archive_prefix_kernel/`. | |
| | **rbf_control infra built and validated (2026-09-04)**: `losses/pqk.py` (`KernelProjectionHead`, trace-normalized Gram alignment loss, gamma ramp, Trap #3 concentration check — kernel-agnostic, shared with `pqk` once it lands) + `quantum/classical_ctrl.py` (RBF Gram matrix, median-heuristic bandwidth). Wired into `trainer.py`: per-side projection heads sized from a real batch (dims differ per architecture, e.g. lenet5's 64 vs resnet50's 2048), trained jointly with the student. Found and fixed a real bug in the same pass — the gamma ramp (0→1 over 10 epochs) was corrupting `small_batch_regime`'s val_loss-based checkpoint selection (Trap #11): val_loss rose every epoch purely because gamma was still ramping, not because the model got worse, so the *first*, least-trained epoch always looked best (rps_25/lenet5 test macro-F1 0.1667 — a random-guess collapse). Fixed by scoring validation at a fixed `gamma_max` regardless of the epoch's ramped training gamma. Kernel concentration (Trap #3) is healthy everywhere tried — off-diagonal K_ij ~0.55-0.63, nowhere near the 0.05 vacuous floor. **rbf_control baseline, lenet5/rps_25 (3 seeds, post-fix)**: macro-F1 0.5892±0.1524 (seeds 0.4136/0.6661/0.6878) — this is the number `pqk` needs to beat by >0.5pt once Stage 4's quantum kernel lands. **rbf_control baseline, trashnet/mobilenetv2_035 (3 seeds)**: macro-F1 0.6672±0.0082 (seeds 0.6713/0.6577/0.6725) — tight variance, sits just below `scratch` (0.6838±0.0141) and `kd` (0.7085±0.0131), the second reference number `pqk` needs to beat by >0.5pt for G2's "2 students" requirement. | |
| | **Quantum infra built (2026-09-04)**: `quantum/kernels.py` — a fixed (non-trainable), `depth`-layer data-re-uploading feature map (RY(angle) per qubit + a CNOT entangling ring, repeated `depth` times) on `lightning.qubit`, extracting each sample's single-qubit Bloch vectors (⟨X⟩,⟨Y⟩,⟨Z⟩ per qubit) via expectation values only — never the full 2^n statevector, which is what makes `diff_method="adjoint"` valid at all (adjoint doesn't support `qml.state()`) and keeps the kernel classically tractable (CLAUDE.md's "projected" in projected quantum kernel). `pqk_gram_matrix` builds `K_ij = exp(-λ·Σ_q‖ρ_q(x_i)-ρ_q(x_j)‖²_F)` from these Bloch vectors using the identity Σ_q‖ρ_q(x_i)-ρ_q(x_j)‖²_F = 0.5·‖bloch(x_i)-bloch(x_j)‖² (single-qubit density matrices ρ=½(I+r·σ), trace(σ_aσ_b)=2δ_ab). Benchmarked batch=128 (Trap #2) on 8 qubits: ~0.27s forward+backward with broadcasting — adds a few minutes per 60-epoch run, not hours. **Barren-plateau check (Trap #7, `scripts/gradient_variance.py`)**: swept n_qubits∈{2,4,6,8,10,12} at depth=4, 100 random-angle samples each, measuring variance of d⟨Z_0⟩/d(angle_0) — variance stayed flat (0.75-1.5) with no exponential decay, so `n_qubits=8` is safe at this depth/entanglement/observable-locality (shallow circuits with local observables are known to resist barren plateaus even as qubit count grows). `pqk` wired into `trainer.py` (`_kernel_loss_and_stats`) and smoke-tested on lenet5/rps_25/seed0 — kernel concentration ~0.94-1.0 offdiag (Trap #3 clear), loss/F1 move sensibly. 3-seed `pqk` vs `rbf_control` comparison (the actual G2 gate) not yet run. | |
| | **G2, first matched pair — lenet5/rps_25, 3 seeds, post-Trap-#13 pipeline (2026-09-05)**. Both twins re-run from scratch on the fixed pipeline (rule #2 — a pipeline change invalidates both, not just `pqk`). macro-F1: `pqk` **0.6375±0.0699** vs `rbf_control` **0.5377±0.2020** → **+9.98pt on means**, clearing G2's ">0.5% macro-F1" bar on this student. **But it is not a clean pass, for three reasons.** (a) Paired by seed, `pqk` wins only 2/3 (seed0 +0.2456, seed1 +0.0985, seed2 −0.0446); the paired difference is +0.0998±0.1451, std larger than the mean — suggestive, not significant at n=3. (b) G2 requires **2 students**; only this one is complete. (c) Most important: on rps_25's *own* primary metric (top-1, per its dataset config) neither kernel method beats plain `scratch` — `scratch` 0.6756±0.1094, `pqk` 0.6425±0.0582, `rbf_control` 0.5842±0.1525. So `pqk` beats the twin it is *required* to beat while both still trail the no-teacher baseline on this pair, which is consistent with lenet5/rps_25 already being logged as INCONCLUSIVE at G1. **The one robust signal is variance, not mean**: `pqk` is the tightest of all four methods here (top-1 std 0.058 vs `scratch` 0.109, `rbf_control` 0.153, and `kd`'s 0.0164-to-0.108 range) — on a pair whose defining problem is seed noise, the quantum kernel behaves like a variance-reducing regularizer even where it doesn't raise the mean. That is the claim the data currently supports; a mean-performance win is not. | |
| | **G2, second pair — trashnet/mobilenetv2_035, 2 of 3 seeds, post-Trap-#13 (2026-09-05)**. macro-F1: `pqk` **0.6959±0.0061** vs `rbf_control` **0.6678±0.0126** → **+2.81pt**, and unlike the rps pair `pqk` wins **both** seeds outright (s0 +0.0413, s1 +0.0149) with the tighter variance again. Against the frozen Stage-1 numbers on this pair (`scratch` 0.6838±0.0141, `kd` 0.7085±0.0131): `pqk` **beats `scratch` by +1.2pt** — the first time either kernel method has cleared the no-teacher baseline — while still trailing `kd` by 1.3pt, and `rbf_control` (0.6678) sits below `scratch`. So on the primary dataset with the primary edge target, the quantum kernel is doing real work that its matched classical twin is not. **Seed 2 was not run** (see cost note below), so this pair is 2/3 and cannot close the gate. **G2 remains OPEN**: it needs 2 students × 3 seeds; we have one student at 3 seeds passing on means only (loses a seed), and one student at 2 seeds passing cleanly. The honest read across both pairs is consistent — `pqk` reliably beats the twin it is required to beat, and reliably has lower seed variance, but it has not yet beaten `kd` anywhere. *Process note:* seed 2 was lost to an orchestration bug of mine, not to the science — a queue script skipped a seed only when **both** twins had results, so when `rbf_control` s1 finished before `pqk` s1 it relaunched the pair, running a duplicate `pqk` s1 concurrently for ~3h. Skip checks must be per-run, not per-pair. | |
| | 🔴 **G2 REVERSES ON THE FULL GRID — treat as a NEGATIVE RESULT (2026-09-14)**. With all 6 pqk/rbf_control pairs complete (69/69 grid runs, 0 failures), `pqk` beats its matched twin on **2 of 6**: trashnet/mnv2_035 **+0.0230** (3/3) and rps_25/lenet5 **+0.0998** (2/3) pass; trashnet/mnv3_small **−0.0419** (1/3), rps_25/mnv3_small **−0.0476** (1/3), rps_25/mnv2_035 **−0.0280** (1/3) and trashnet/lenet5 **−0.0042** (2/3) all fail. **The two passing pairs are exactly the two the gate was originally evaluated on** — the "PASS" recorded on 2026-09-13 was a selection effect, not a result. A 2-of-6 win rate with per-seed wins of 3/3, 2/3, 2/3, 1/3, 1/3, 1/3 is what noise looks like; there is no consistent quantum advantage here. Per this table's own "if it fails" column, the project should now be written up as **a negative result + edge benchmark**. **The genuinely interesting finding is the control, not the treatment**: `rbf_control` — classical RBF kernel alignment — is strong in its own right, beating `scratch` on 4 of 6 pairs and beating *both* `scratch` and `kd` on rps_25/mnv3_small (0.8877 vs 0.8157/0.8296) and on trashnet/lenet5 (0.5392 vs 0.5286/0.4838). So *kernel-alignment distillation works*; swapping the classical kernel for a projected quantum kernel is what fails to add value. That is a cleaner and more defensible paper than the original thesis. ⚠️ **Methodological warning for any re-tune**: λ and the un-tuned τ=0.5 are the obvious levers, but tuning them against these same test numbers until `pqk` wins would be fitting the test set (garden of forking paths). Any retune must be screened on validation/seed-0 of a *single* pair and then confirmed on the other five untouched — and the headline number must come from that pre-registered protocol, not from the best-looking cell. | |
| | **rps_25/lenet5 — accepted FAIL, not pursued (2026-09-14, user decision)**. On this pair `scratch` (0.6764±0.1084) beats *every* distillation method tried: `kd` 0.6502 (−0.0262), `rkd` failed badly at both weight scales (0.3297 / 0.1889), `rbf_control` 0.5377 (−0.1387), `pqk` 0.6375 (−0.0388). Adaptive λ made `pqk` *worse* (seed-0 0.4343). The ceiling is structural, not a `pqk` defect: `pqk` already runs this pair's own tuned τ=0.5 (inherited from `kd_lenet5_rps25`), and lowering γ only collapses `pqk` toward `kd`'s 0.6502 — still under the bar. Clearing `scratch` here would require the kernel to add something beyond `kd`, on 630 images with a ~35K-param student, and nothing in the data suggests it does. Note also that `scratch`'s mean rests on one lucky seed (0.5564/0.7054/0.7673), so "beating" it is substantially a coin toss. Left as a documented FAIL rather than tuned until some λ/seed combination clears it — that would be fitting the test set and would not replicate. Consistent with this pair already being logged INCONCLUSIVE at G1. | |
| | **trashnet/mobilenetv3_small — FAIL after a 5-config screen (2026-09-14)**. Seed-0 screens vs `scratch` 0.7236: adaptive λ **0.7081 (−0.0155, best)**, adaptive+τ=3 0.7065, fixed λ 0.6998, adaptive+depth=2 0.6881, adaptive+γ=0.5 0.6855, adaptive+γ=2.0 0.6790. The *unmodified* adaptive-λ config is the best of the six — both γ directions and the depth change made it worse, so this is a local optimum, not an untuned setup. Seed 0 also runs optimistic on this pair (fixed λ scored 0.6998 at s0 but 0.6733 over 3 seeds), so the real gap is nearer −0.04. Stopped at five candidates per screen-then-stop. **The informative part is that this is not a `pqk` problem**: on this student the *locked* `rbf_control` also fails (−0.0084) while `kd` clears `scratch` (+0.0089) and `rkd` is level (−0.0081). So it is **kernel-alignment distillation as a family** — classical and quantum alike — that underperforms on mobilenetv3_small. Leading hypothesis: this is the one student carrying `force_batch_stats_only` (Trap #10), and a batch-Gram alignment loss and per-batch BatchNorm statistics are computed over the *same* batch, so unstable statistics feed straight into the kernel. Untested, but it is the one structural difference between this student and the pairs where the kernel methods work, and it predicts the observed classical/quantum symmetry. | |
| | **Two-body correlators (`pqk_zz`) — real improvement, still a FAIL (2026-09-15)**. Motivation was measured, not assumed: the single-qubit-only kernel reads 3n marginals, and those marginals get *less* informative as the circuit grows — mean Bloch length \|r\| falls 0.541 (4 qubits) → 0.384 (8) → 0.313 (12), i.e. each reduced state drifts toward maximally mixed. That is why raising `n_qubits` made things worse, not better (effective-dim ratio fell 1.40 at n=4 to 0.87 at n=12, for 2.5× the compute). Adding the n(n−1)/2 ⟨Z_iZ_j⟩ correlators restores the two-body information: at 8 qubits the feature goes 24 → 52 dims, effective dimensionality 8.90 → 10.92, ratio 1.11 → 1.36, for ~25% more simulator time and still polynomially many observables. **On trashnet/mobilenetv3_small (3 seeds): `pqk_zz` 0.7092±0.0327 vs `pqk` fixed-λ 0.6733±0.0235 — a genuine +3.6pt, closing the gap to `scratch` from −0.0503 to −0.0143 and making it the best `pqk` variant on this pair. But it still FAILS** (`scratch` 0.7236, `rbf_control` 0.7152). **Methodological note worth keeping**: the seed-0 screen scored 0.7312 and appeared to clear `scratch` by +0.0076 — that was predicted in advance to be seed-0 optimism, since fixed λ had shown a +0.027 s0-vs-3-seed bias on this exact pair. Actual bias: **+0.0220** (0.7312 → 0.7092). The prediction was made before the confirmation ran, and the margin was correctly judged to sit inside the bias. Screens on this pair must not be believed under ~+0.03. Note also `pqk_zz` *regressed* on a pair where `pqk` was passing (rps_25/mobilenetv2_035, seed-0 0.6330 vs adaptive-λ 0.7153) — like λ, correlators are a per-pair trade, not a universal upgrade. | |
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
10. RPS's 630-image train set is only ~5 batches/epoch at batch 128 — too few for a
    from-scratch student's BatchNorm running stats to stabilize. Symptom: train_f1
    improves normally while val_f1 freezes at an exact constant every epoch (the
    macro-F1 of a model that always predicts one class) — train mode looks fine
    (per-batch stats) but eval mode degenerates (garbage running stats). The
    pretrained teacher doesn't hit this (already-good BN stats before fine-tuning);
    students trained on rps_25 disable running-stat tracking instead
    (`use_batch_stats_only`, gated by `dataset.small_batch_regime`). Same fix, same
    symptom, different cause: `mobilenetv3_small` shows this on *trashnet* too (14
    batches/epoch — not a small-batch dataset) — its Hardswish/Hardsigmoid (SE-block)
    activations destabilize BN from random init regardless of batch count. Gating is
    now `dataset.small_batch_regime OR student.force_batch_stats_only` (see
    `configs/student/mobilenetv3_small.yaml`), and the calibration loader
    (`build_bn_calibration_loader`) is dataset-agnostic — built from whatever
    `train_loader` the dataset produced instead of being hardcoded to rps_25's split.
11. RPS's `validation/` split is only 33 images (11/class, vs. a 372-image `test/`) —
    val_f1 moves in ~3pt steps and swings wildly epoch to epoch even after the model has
    converged (observed on kd/seed0: 0.53→0.31→0.64→0.82→0.73 while val_loss fell
    smoothly and monotonically the whole time). Selecting the best checkpoint by val_f1
    picks a lucky spike rather than the most-converged model — kd/rkd scored *below*
    `scratch` on rps_25 test purely from bad epoch selection (test acc 0.586/0.640 vs
    scratch's 0.707), despite kd/rkd having the higher best_val_f1. Fix: checkpoint
    selection uses val_loss instead of val_f1 whenever `dataset.small_batch_regime` is
    set (`trainer.py`, gated the same way as Trap #10's BN fix) — loss is continuous and
    far less noisy on a 33-image sample.
12. Trap #11's val_loss fix breaks again once a loss term has a training-time ramp
    (`rbf_control`/`pqk`'s gamma, ramped 0→`gamma_loss_weight` over `gamma_ramp_epochs`):
    comparing val_loss across epochs while its own weighting is still changing means the
    *ramp*, not the model, drives val_loss up every epoch — the least-trained checkpoint
    (epoch 1) always looks best. Symptom on rps_25/lenet5/`rbf_control`: val_loss rose
    monotonically every single epoch despite val_f1 recovering to 0.52 by epoch 13, and
    the selected checkpoint scored test macro-F1 0.1667 (a random-guess collapse). Fix:
    score the validation pass at a fixed `gamma_max`, not the epoch's ramped `gamma`
    (`trainer.py`) — the ramp still applies to the training-time gradient step, it just
    can't be allowed to contaminate the metric used to compare checkpoints across epochs.
13. **Mutual collapse is a global optimum of the kernel-alignment loss** if the
    teacher-side projection head is trainable. `‖K̃_t − K̃_s‖²_F` is driven to *exactly*
    zero by collapsing both projections to constant maps: every Gram entry goes to 1,
    trace-normalization makes both matrices identical, loss = 0, and the student has
    learned nothing. Discovered 2026-09-05 from `pqk`'s logged concentration —
    `kernel_offdiag_student=0.9953, teacher=0.9864` at test time. The teacher backbone is
    frozen and its features genuinely vary, so a teacher-side off-diagonal of 0.986 can
    only mean its projection head had collapsed to a near-constant map (for reference,
    random spread-out angles at λ=1 give off-diagonal mean 0.37, std 0.16). Fix: the
    teacher-side projection is fixed at random init, excluded from the optimizer, and
    evaluated under `no_grad` — a random linear map preserves the teacher's relational
    structure well enough to be the target kernel (the same thing `rkd` does with raw
    teacher features), and freezing it is what keeps that target non-degenerate.
    Two things made this easy to miss: (a) Trap #3's check only guarded the *low* end
    (`K_ij < 0.05`, "every pair looks unique") and was blind to the saturated end
    (`K_ij → 1`, "every pair looks alike") — equally vacuous, now warned on at >0.95;
    (b) `rbf_control` is *immune* to the saturation symptom because its median-heuristic
    bandwidth rescales to the data spread every batch, pinning off-diagonal ≈0.60 no
    matter how collapsed the projection is. So the classical twin looked perfectly
    healthy while sharing the identical latent flaw. Per rule #2 the twins must share a
    pipeline, so fixing this invalidates *both* methods' earlier numbers, not just
    `pqk`'s — pre-fix results archived under `archive_prefix_kernel/`.

## Commands

```bash
python -m qakd.train.teacher teacher=resnet50_in1k dataset=trashnet     # once per dataset, idempotent

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
| 2 | Stage 0/1 methods | `scratch`, `kd`, `rkd` running end-to-end on ≥1 student × both datasets, 3 seeds | **Done** — frozen (2026-08-29): all 3 students × both datasets confirmed over 3 seeds; see Go/no-go |
| 3 | **G1 gate** | `kd` beats `scratch` by a clear margin on trashnet — else fix the pipeline before going further | **Passed** — see Go/no-go |
| 4 | Quantum infra | `quantum/kernels.py` (`lightning.qubit`, 8-qubit encoding, batch Gram matrix), `quantum/classical_ctrl.py` (RBF control), `scripts/gradient_variance.py` run for 2–12 qubits | **Done** (2026-09-04) — see Go/no-go |
| 5 | Stage 2 methods | `rbf_control` + `pqk` running on lenet5/rps_25, kernel concentration logged | **Done** (2026-09-05) — both methods run end-to-end on the post-Trap-#13 pipeline, concentration logged and now range-checked at both ends |
| 6 | **G2 gate (week 2)** | `pqk` beats `rbf_control` by >0.5% macro-F1, 2 students, 3 seeds — else write up as a negative result + edge benchmark only | **FAILED on the full grid** (2026-09-14) — the 09-13 "pass" held only on the 2 pairs then tested; across all 6 pairs `pqk` wins 2/6. Take the "if it fails" branch: negative result + edge benchmark. See Go/no-go |
| 7 | Full sweep | Remaining runs from the 83-run grid (or the 44-run minimum-viable set) — `results/{run_id}/`, immutable | **Done** (2026-09-14) — **69/69** grid runs present, 0 failures (ablation block dropped by user decision, so the plan is 71 incl. teachers). `results.json` holds 87 student runs incl. out-of-grid extras |
| 8 | Deploy pipeline | `deploy/export.py` (ONNX opset 17) + `deploy/purity_check.py` + TFLite INT8 per-channel + parity check | Not started |
| 9 | **G3 gate** | `pqk` INT8 beats `kd` INT8 at matched KB — else drop the edge claim from the title | Not started |
| 10 | Edge benchmark | `edge/bench/run.py` on Raspberry Pi 4B, Jetson Orin Nano, ESP32-S3 (lenet5/mobilenetv2_035 only) — latency, mJ/inference, SoC temp | Not started |
| 11 | Aggregation | `scripts/aggregate.py` — final results table (macro-F1, top-1, params, INT8 KB, MACs, latency, mJ, ECE) across all methods × students × datasets | Not started |

## Out of scope

Quantum ops at inference. Quantum advantage claims. >12 qubits. Federated quantum training.
