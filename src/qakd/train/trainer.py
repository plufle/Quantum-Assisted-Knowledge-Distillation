import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW

import qakd.models.teachers  # noqa: F401 — registers TEACHERS entries on import
from qakd.data.datasets import build_rps25_loaders, build_trashnet_loaders
from qakd.exception import CustomException
from qakd.logger import logger
from qakd.models.registry import build_teacher
from qakd.utils import set_seed

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
