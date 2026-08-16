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
from qakd.data.datasets import build_rps25_loaders, build_trashnet_loaders
from qakd.exception import CustomException
from qakd.logger import logger
from qakd.losses.kd import kd_loss
from qakd.losses.rkd import rkd_loss
from qakd.models.common import use_batch_stats_only
from qakd.models.registry import build_student, build_teacher
from qakd.utils import set_seed

# Config keys on `cfg.student` that are training recipe, not model constructor kwargs.
_STUDENT_RECIPE_KEYS = {"name", "epochs", "patience", "optimizer", "lr", "weight_decay"}

_LOADER_BUILDERS = {
    "trashnet": build_trashnet_loaders,
    "rps_25": build_rps25_loaders,
}


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


def _student_loss(method_cfg, model, teacher, images, labels):
    if method_cfg.name == "scratch":
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
    elif method_cfg.name == "kd":
        logits = model(images)
        with torch.no_grad():
            teacher_logits = teacher(images)
        loss = kd_loss(logits, teacher_logits, labels, method_cfg.loss.alpha, method_cfg.loss.beta, method_cfg.loss.tau)
    elif method_cfg.name == "rkd":
        logits, student_feats = model(images, return_feats=True)
        with torch.no_grad():
            _, teacher_feats = teacher(images, return_feats=True)
        loss = rkd_loss(
            student_feats[-1], teacher_feats[-1], logits, labels,
            method_cfg.loss.ce_weight, method_cfg.loss.distance_weight, method_cfg.loss.angle_weight,
        )
    else:
        raise ValueError(f"Stage 2 doesn't implement method={method_cfg.name!r} (pqk/rbf_control land in Stage 5)")
    return logits, loss


def _run_student_epoch(model, teacher, loader, device, method_cfg, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    all_preds, all_labels = [], []
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(train_mode):
            logits, loss = _student_loss(method_cfg, model, teacher, images, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    top1_acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(all_labels), macro_f1, top1_acc


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

        student_kwargs = {k: v for k, v in cfg.student.items() if k not in _STUDENT_RECIPE_KEYS}
        model = build_student(student_name, num_classes=cfg.dataset.num_classes, **student_kwargs)
        if cfg.dataset.get("small_batch_regime", False):  # Trap #10
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

        optimizer = AdamW(model.parameters(), lr=cfg.student.lr, weight_decay=cfg.student.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.student.epochs)

        best_val_f1, best_state, epochs_without_improvement = -1.0, None, 0
        for epoch in range(cfg.student.epochs):
            train_loss, train_f1, train_acc = _run_student_epoch(
                model, teacher, train_loader, device, cfg.method, optimizer
            )
            val_loss, val_f1, val_acc = _run_student_epoch(model, teacher, val_loader, device, cfg.method)
            scheduler.step()
            logger.info(
                "[%s] epoch %d/%d train_loss=%.4f train_f1=%.4f val_loss=%.4f val_f1=%.4f",
                run_id, epoch + 1, cfg.student.epochs, train_loss, train_f1, val_loss, val_f1,
            )
            if val_f1 > best_val_f1:
                best_val_f1, best_state, epochs_without_improvement = val_f1, copy.deepcopy(model.state_dict()), 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.student.patience:
                    logger.info("[%s] early stopping at epoch %d.", run_id, epoch + 1)
                    break

        model.load_state_dict(best_state)
        _, test_f1, test_acc = _run_student_epoch(model, teacher, test_loader, device, cfg.method)

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
            "recipe": {
                "epochs_ran": epoch + 1,
                "epochs_budget": cfg.student.epochs,
                "patience": cfg.student.patience,
                "optimizer": cfg.student.optimizer,
                "lr": cfg.student.lr,
                "weight_decay": cfg.student.weight_decay,
                "batch_size": cfg.dataset.batch_size,
                "method_loss": OmegaConf.to_container(cfg.method.loss) if "loss" in cfg.method else None,
            },
            "git_sha": _git_sha(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved results to results/%s/metrics.json (test macro-F1=%.4f)", run_id, test_f1)
        return metrics
    except Exception as e:
        raise CustomException(e, sys) from e
