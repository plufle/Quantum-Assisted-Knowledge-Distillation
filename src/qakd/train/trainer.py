import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW

import qakd.models.students  # noqa: F401 — registers STUDENTS entries on import
import qakd.models.teachers  # noqa: F401 — registers TEACHERS entries on import
from qakd.data.datasets import (
    build_bn_calibration_loader,
    build_rps25_loaders,
    build_trashnet_loaders,
)
from qakd.exception import CustomException
from qakd.logger import logger
from qakd.losses.kd import kd_loss
from qakd.losses.pqk import KernelProjectionHead, gamma_ramp, kernel_alignment_loss, mean_offdiagonal
from qakd.losses.rkd import rkd_loss
from qakd.models.common import calibrate_batch_norm, use_batch_stats_only
from qakd.models.registry import build_student, build_teacher
from qakd.quantum.classical_ctrl import rbf_gram_matrix
from qakd.quantum.kernels import pqk_gram_matrix
from qakd.utils import set_seed

_KERNEL_METHODS = {"rbf_control", "pqk"}

# Config keys on `cfg.student` that are training recipe, not model constructor kwargs.
_STUDENT_RECIPE_KEYS = {
    "name", "epochs", "patience", "optimizer", "lr", "weight_decay", "force_batch_stats_only",
}

_LOADER_BUILDERS = {
    "trashnet": build_trashnet_loaders,
    "rps_25": build_rps25_loaders,
}


def _balanced_class_weights(entries, num_classes, device):
    counts = [0] * num_classes
    for entry in entries:
        counts[entry["label"]] += 1
    total = sum(counts)
    weights = [total / (num_classes * count) if count > 0 else 0.0 for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    all_preds, all_labels = [], []
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(all_labels), macro_f1


def train_teacher(cfg):
    """Fine-tunes `cfg.teacher` on `cfg.dataset`. Idempotent: if a checkpoint + sidecar
    already exist for this (teacher, dataset) pair, they're loaded and returned as-is —
    teachers are trained once per dataset (CLAUDE.md)."""
    dataset_name = cfg.dataset.name
    teacher_name = cfg.teacher.name
    ckpt_dir = "checkpoints/teachers"
    ckpt_path = os.path.join(ckpt_dir, f"{teacher_name}__{dataset_name}.pt")
    sidecar_path = os.path.join(ckpt_dir, f"{teacher_name}__{dataset_name}.json")

    if os.path.exists(ckpt_path) and os.path.exists(sidecar_path):
        logger.info("Teacher checkpoint already exists at %s — skipping training.", ckpt_path)
        with open(sidecar_path) as f:
            return json.load(f)

    try:
        set_seed(cfg.seed)
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        build_loaders = _LOADER_BUILDERS[dataset_name]
        train_loader, val_loader, test_loader = build_loaders(cfg.dataset)

        model = build_teacher(
            teacher_name, num_classes=cfg.dataset.num_classes, pretrained=cfg.teacher.pretrained
        ).to(device)
        params = _count_params(model)
        logger.info("%s has %d parameters", teacher_name, params)
        optimizer = AdamW(model.parameters(), lr=cfg.teacher.lr, weight_decay=cfg.teacher.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.teacher.epochs)

        best_val_f1, best_state, epochs_without_improvement = -1.0, None, 0
        for epoch in range(cfg.teacher.epochs):
            train_loss, train_f1 = _run_epoch(model, train_loader, device, optimizer)
            val_loss, val_f1 = _run_epoch(model, val_loader, device)
            scheduler.step()
            logger.info(
                "epoch %d/%d train_loss=%.4f train_f1=%.4f val_loss=%.4f val_f1=%.4f",
                epoch + 1, cfg.teacher.epochs, train_loss, train_f1, val_loss, val_f1,
            )
            if val_f1 > best_val_f1:
                best_val_f1, best_state, epochs_without_improvement = val_f1, copy.deepcopy(model.state_dict()), 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.teacher.patience:
                    logger.info("Early stopping at epoch %d (no val improvement for %d epochs).",
                                epoch + 1, cfg.teacher.patience)
                    break

        model.load_state_dict(best_state)
        _, test_f1 = _run_epoch(model, test_loader, device)

        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        sidecar = {
            "teacher": teacher_name,
            "dataset": dataset_name,
            "macro_f1": test_f1,
            "best_val_macro_f1": best_val_f1,
            "params": params,
            "recipe": {
                "epochs_ran": epoch + 1,
                "epochs_budget": cfg.teacher.epochs,
                "patience": cfg.teacher.patience,
                "optimizer": cfg.teacher.optimizer,
                "lr": cfg.teacher.lr,
                "weight_decay": cfg.teacher.weight_decay,
                "batch_size": cfg.dataset.batch_size,
                "seed": cfg.seed,
                "pretrained": cfg.teacher.pretrained,
            },
            "git_sha": _git_sha(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2)
        logger.info("Saved teacher checkpoint to %s (test macro-F1=%.4f)", ckpt_path, test_f1)
        return sidecar
    except Exception as e:
        raise CustomException(e, sys) from e


def _penultimate_dim(model, sample_images, device):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, feats = model(sample_images.to(device), return_feats=True)
    model.train(was_training)
    return feats[-1].shape[1]


def _kernel_loss_and_stats(method_cfg, kernel_modules, student_feats, teacher_feats, gamma):
    student_proj = kernel_modules["student_proj"](student_feats)
    # Trap #13: the teacher side is a FIXED random projection evaluated under no_grad —
    # it is the target the student matches, never a party to the optimization. Letting it
    # train (as this originally did) makes mutual collapse a global optimum of the
    # alignment loss: drive both projections to constants, both Gram matrices go to
    # all-ones, trace-normalize to the same matrix, and the loss hits exactly 0 while the
    # student has learned nothing.
    with torch.no_grad():
        teacher_proj = kernel_modules["teacher_proj"](teacher_feats.detach())
    if method_cfg.kernel == "rbf":
        bandwidth = method_cfg.get("bandwidth", None)
        student_gram = rbf_gram_matrix(student_proj, bandwidth)
        teacher_gram = rbf_gram_matrix(teacher_proj, bandwidth)
    elif method_cfg.kernel == "projected_quantum":
        student_gram = pqk_gram_matrix(
            student_proj, method_cfg.n_qubits, method_cfg.depth, method_cfg.lambda_, method_cfg.device_backend
        )
        with torch.no_grad():
            teacher_gram = pqk_gram_matrix(
                teacher_proj, method_cfg.n_qubits, method_cfg.depth, method_cfg.lambda_, method_cfg.device_backend
            )
    else:
        raise ValueError(f"Unknown kernel={method_cfg.kernel!r}")
    kernel_term = gamma * kernel_alignment_loss(teacher_gram, student_gram)
    stats = {
        "kernel_offdiag_student": mean_offdiagonal(student_gram).item(),
        "kernel_offdiag_teacher": mean_offdiagonal(teacher_gram).item(),
    }
    return kernel_term, stats


def _student_loss(
    method_cfg,
    model,
    teacher,
    images,
    labels,
    ce_class_weights=None,
    ce_label_smoothing=0.0,
    kernel_modules=None,
    gamma=0.0,
):
    aux = {}
    if method_cfg.name == "scratch":
        logits = model(images)
        loss = nn.functional.cross_entropy(
            logits,
            labels,
            weight=ce_class_weights,
            label_smoothing=ce_label_smoothing,
        )
    elif method_cfg.name == "kd":
        logits = model(images)
        with torch.no_grad():
            teacher_logits = teacher(images)
        loss = kd_loss(
            logits,
            teacher_logits,
            labels,
            method_cfg.loss.alpha,
            method_cfg.loss.beta,
            method_cfg.loss.tau,
            ce_class_weights=ce_class_weights,
            ce_label_smoothing=ce_label_smoothing,
        )
    elif method_cfg.name == "rkd":
        logits, student_feats = model(images, return_feats=True)
        with torch.no_grad():
            _, teacher_feats = teacher(images, return_feats=True)
        loss = rkd_loss(
            student_feats[-1],
            teacher_feats[-1],
            logits,
            labels,
            method_cfg.loss.ce_weight,
            method_cfg.loss.distance_weight,
            method_cfg.loss.angle_weight,
            ce_class_weights=ce_class_weights,
            ce_label_smoothing=ce_label_smoothing,
        )
    elif method_cfg.name in _KERNEL_METHODS:
        logits, student_feats = model(images, return_feats=True)
        with torch.no_grad():
            teacher_logits, teacher_feats = teacher(images, return_feats=True)
        base_loss = kd_loss(
            logits,
            teacher_logits,
            labels,
            method_cfg.loss.alpha,
            method_cfg.loss.beta,
            method_cfg.loss.tau,
            ce_class_weights=ce_class_weights,
            ce_label_smoothing=ce_label_smoothing,
        )
        kernel_term, aux = _kernel_loss_and_stats(
            method_cfg, kernel_modules, student_feats[-1], teacher_feats[-1], gamma
        )
        loss = base_loss + kernel_term
    else:
        raise ValueError(f"Unknown method={method_cfg.name!r}")
    return logits, loss, aux


def _run_student_epoch(
    model,
    teacher,
    loader,
    device,
    method_cfg,
    optimizer=None,
    ce_class_weights=None,
    ce_label_smoothing=0.0,
    kernel_modules=None,
    gamma=0.0,
):
    train_mode = optimizer is not None
    model.train(train_mode)
    all_preds, all_labels = [], []
    total_loss = 0.0
    kernel_stats_sum, kernel_stats_count = {}, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(train_mode):
            logits, loss, aux = _student_loss(
                method_cfg,
                model,
                teacher,
                images,
                labels,
                ce_class_weights=ce_class_weights,
                ce_label_smoothing=ce_label_smoothing,
                kernel_modules=kernel_modules,
                gamma=gamma,
            )
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        for key, value in aux.items():
            kernel_stats_sum[key] = kernel_stats_sum.get(key, 0.0) + value
        if aux:
            kernel_stats_count += 1
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    top1_acc = accuracy_score(all_labels, all_preds)
    kernel_stats = (
        {key: value / kernel_stats_count for key, value in kernel_stats_sum.items()} if kernel_stats_count else {}
    )
    return total_loss / len(all_labels), macro_f1, top1_acc, kernel_stats


def train_student(cfg):
    """Trains `cfg.student` on `cfg.dataset` with `cfg.method`. Idempotent: results/{run_id}/
    is immutable — if metrics.json already exists there, it's loaded and returned as-is,
    never overwritten (CLAUDE.md)."""
    dataset_name = cfg.dataset.name
    student_name = cfg.student.name
    method_name = cfg.method.name
    seed = cfg.seed
    run_id = f"{dataset_name}__{student_name}__{method_name}__s{seed}"
    results_dir = os.path.join("results", run_id)
    metrics_path = os.path.join(results_dir, "metrics.json")

    if os.path.exists(metrics_path):
        logger.info("results/%s/metrics.json already exists — skipping (results/ is immutable).", run_id)
        with open(metrics_path) as f:
            return json.load(f)

    try:
        set_seed(seed)
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        build_loaders = _LOADER_BUILDERS[dataset_name]
        train_loader, val_loader, test_loader = build_loaders(cfg.dataset)
        # Trap #10 originally covered only rps_25's small_batch_regime (too few batches/epoch
        # to stabilize BatchNorm running stats). `force_batch_stats_only` extends the same
        # fix to specific students that show the identical eval-mode collapse elsewhere —
        # e.g. mobilenetv3_small on trashnet, whose Hardswish/Hardsigmoid (SE-block) gates
        # produce unstable BN statistics from random init regardless of batch count.
        needs_batch_stats_only = cfg.dataset.get("small_batch_regime", False) or cfg.student.get(
            "force_batch_stats_only", False
        )
        bn_calibration_loader = (
            build_bn_calibration_loader(cfg.dataset, train_loader) if needs_batch_stats_only else None
        )
        ce_class_weights = None
        if cfg.dataset.get("class_weighted_ce", False):
            ce_class_weights = _balanced_class_weights(
                train_loader.dataset.entries, cfg.dataset.num_classes, device
            )
        ce_label_smoothing = float(cfg.dataset.get("ce_label_smoothing", 0.0))

        student_kwargs = {k: v for k, v in cfg.student.items() if k not in _STUDENT_RECIPE_KEYS}
        model = build_student(student_name, num_classes=cfg.dataset.num_classes, **student_kwargs)
        params = _count_params(model)
        logger.info("%s has %d parameters", student_name, params)
        if needs_batch_stats_only:  # Trap #10
            model = use_batch_stats_only(model)
        model = model.to(device)

        teacher = None
        if method_name != "scratch":
            teacher_ckpt = os.path.join("checkpoints", "teachers", f"{cfg.teacher.name}__{dataset_name}.pt")
            if not os.path.exists(teacher_ckpt):
                raise FileNotFoundError(
                    f"Teacher checkpoint not found at {teacher_ckpt} — run "
                    f"`python -m qakd.train.teacher dataset={dataset_name}` first (Stage 1)."
                )
            teacher = build_teacher(cfg.teacher.name, num_classes=cfg.dataset.num_classes, pretrained=False)
            teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
            teacher = teacher.to(device)
            teacher.eval()
            for p in teacher.parameters():  # Trap #6
                p.requires_grad_(False)

        kernel_modules = None
        trainable_params = list(model.parameters())
        if method_name in _KERNEL_METHODS:
            # Projection heads map each side's own penultimate feature dim down to
            # `n_qubits` (CLAUDE.md Loss) — dims differ per architecture (e.g. lenet5's
            # 64 vs resnet50's 2048), so infer them from one real batch rather than
            # hardcoding per-model shapes.
            sample_images, _ = next(iter(train_loader))
            student_dim = _penultimate_dim(model, sample_images, device)
            teacher_dim = _penultimate_dim(teacher, sample_images, device)
            student_proj = KernelProjectionHead(student_dim, cfg.method.n_qubits).to(device)
            teacher_proj = KernelProjectionHead(teacher_dim, cfg.method.n_qubits).to(device)
            # Teacher-side projection stays fixed at its random init and is NOT added to
            # the optimizer (Trap #13) — a random linear map preserves the teacher's
            # relational structure well enough to serve as the target kernel (the same
            # thing rkd does with raw teacher features), and freezing it is what makes
            # that target non-degenerate.
            teacher_proj.eval()
            for p in teacher_proj.parameters():
                p.requires_grad_(False)
            kernel_modules = {"student_proj": student_proj, "teacher_proj": teacher_proj}
            trainable_params += list(student_proj.parameters())

        optimizer = AdamW(trainable_params, lr=cfg.student.lr, weight_decay=cfg.student.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.student.epochs)
        gamma_max = cfg.method.get("gamma_loss_weight", 0.0) if kernel_modules is not None else 0.0
        gamma_ramp_epochs = cfg.method.get("gamma_ramp_epochs", 0) if kernel_modules is not None else 0
        kernel_concentration_warned = False

        # Trap #11: rps_25's validation split is only 33 images (11/class) — val_f1 moves
        # in ~3pt steps and swings wildly epoch to epoch (observed: 0.53->0.31->0.64->0.82
        # ->0.73 while val_loss fell smoothly and monotonically the whole time), so
        # selecting the best checkpoint by val_f1 picks a lucky spike rather than the most
        # converged model and generalizes badly to the real (372-image) test split.
        # val_loss is continuous and far less noisy on tiny samples, so use it for
        # checkpoint selection in this regime instead.
        select_by_loss = cfg.dataset.get("small_batch_regime", False)
        best_selection_score = float("inf") if select_by_loss else -1.0
        best_val_f1, best_state, epochs_without_improvement = -1.0, None, 0
        for epoch in range(cfg.student.epochs):
            gamma = gamma_ramp(epoch, gamma_ramp_epochs, gamma_max) if kernel_modules is not None else 0.0
            train_loss, train_f1, train_acc, train_kernel_stats = _run_student_epoch(
                model,
                teacher,
                train_loader,
                device,
                cfg.method,
                optimizer,
                ce_class_weights=ce_class_weights,
                ce_label_smoothing=ce_label_smoothing,
                kernel_modules=kernel_modules,
                gamma=gamma,
            )
            if bn_calibration_loader is not None:
                calibrate_batch_norm(model, bn_calibration_loader, device)
            # val_loss is used for checkpoint selection (Trap #11) — evaluate it at a
            # fixed gamma_max rather than the epoch's ramped `gamma`, otherwise the
            # ramp itself (not the model) drives val_loss up every epoch and the
            # earliest, least-trained checkpoint always looks "best".
            val_loss, val_f1, val_acc, val_kernel_stats = _run_student_epoch(
                model,
                teacher,
                val_loader,
                device,
                cfg.method,
                ce_class_weights=ce_class_weights,
                ce_label_smoothing=ce_label_smoothing,
                kernel_modules=kernel_modules,
                gamma=gamma_max,
            )
            if bn_calibration_loader is not None:
                use_batch_stats_only(model)
            scheduler.step()
            kernel_log = ""
            if train_kernel_stats:
                kernel_log = (
                    f" kernel_offdiag(student={train_kernel_stats['kernel_offdiag_student']:.4f},"
                    f" teacher={train_kernel_stats['kernel_offdiag_teacher']:.4f})"
                )
                # Trap #3, both ends: < 0.05 means every pair looks unique, > 0.95 means
                # every pair looks alike (a saturated, all-ones Gram). Either way the
                # alignment loss carries no structure and is vacuous.
                if not kernel_concentration_warned and (
                    min(train_kernel_stats.values()) < 0.05 or max(train_kernel_stats.values()) > 0.95
                ):
                    logger.warning(
                        "[%s] kernel concentration out of range at epoch %d (student=%.4f teacher=%.4f) "
                        "— L_kernel may be vacuous.",
                        run_id, epoch + 1,
                        train_kernel_stats["kernel_offdiag_student"],
                        train_kernel_stats["kernel_offdiag_teacher"],
                    )
                    kernel_concentration_warned = True
            logger.info(
                "[%s] epoch %d/%d train_loss=%.4f train_f1=%.4f val_loss=%.4f val_f1=%.4f%s",
                run_id, epoch + 1, cfg.student.epochs, train_loss, train_f1, val_loss, val_f1, kernel_log,
            )
            current_score = val_loss if select_by_loss else val_f1
            is_better = current_score < best_selection_score if select_by_loss else current_score > best_selection_score
            if is_better:
                best_selection_score = current_score
                best_val_f1, best_state, epochs_without_improvement = val_f1, copy.deepcopy(model.state_dict()), 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.student.patience:
                    logger.info("[%s] early stopping at epoch %d.", run_id, epoch + 1)
                    break

        model.load_state_dict(best_state)
        if bn_calibration_loader is not None:
            calibrate_batch_norm(model, bn_calibration_loader, device)
        _, test_f1, test_acc, test_kernel_stats = _run_student_epoch(
            model,
            teacher,
            test_loader,
            device,
            cfg.method,
            ce_class_weights=ce_class_weights,
            ce_label_smoothing=ce_label_smoothing,
            kernel_modules=kernel_modules,
            gamma=gamma_max,
        )

        ckpt_dir = "checkpoints/students"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{run_id}.pt")
        torch.save(model.state_dict(), ckpt_path)

        os.makedirs(results_dir, exist_ok=True)
        metrics = {
            "run_id": run_id,
            "dataset": dataset_name,
            "student": student_name,
            "method": method_name,
            "seed": seed,
            "macro_f1": test_f1,
            "top1_accuracy": test_acc,
            "best_val_macro_f1": best_val_f1,
            "params": params,
            "checkpoint": ckpt_path,
            "kernel_offdiag_student": test_kernel_stats.get("kernel_offdiag_student") if test_kernel_stats else None,
            "kernel_offdiag_teacher": test_kernel_stats.get("kernel_offdiag_teacher") if test_kernel_stats else None,
            "recipe": {
                "epochs_ran": epoch + 1,
                "epochs_budget": cfg.student.epochs,
                "patience": cfg.student.patience,
                "optimizer": cfg.student.optimizer,
                "lr": cfg.student.lr,
                "weight_decay": cfg.student.weight_decay,
                "batch_size": cfg.dataset.batch_size,
                "batch_norm_calibrated": bn_calibration_loader is not None,
                "class_weighted_ce": ce_class_weights is not None,
                "ce_label_smoothing": ce_label_smoothing,
                "method_loss": OmegaConf.to_container(cfg.method.loss) if "loss" in cfg.method else None,
            },
            "git_sha": _git_sha(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved checkpoint to %s and results to results/%s/metrics.json (test macro-F1=%.4f)",
                    ckpt_path, run_id, test_f1)
        return metrics
    except Exception as e:
        raise CustomException(e, sys) from e
