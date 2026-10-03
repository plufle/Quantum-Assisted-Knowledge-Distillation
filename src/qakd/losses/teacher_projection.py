"""Train-only fitted teacher projection for the kernel-alignment target.

The trainer's default teacher-side projection is a fixed *random* linear map 2048 -> 8 angles
(Trap #13 requires it frozen). Validation diagnostics (scripts/teacher_kernel_diagnostics.py)
showed it discards roughly half the teacher's class structure: centered kernel-target alignment
of the teacher kernel falls from 0.720 (raw 2048-d) to 0.363 (random angles) on trashnet. A
projection *fitted on the training split only* and frozen before student training restores it.

`lda_pca`: the C-1 shrinkage-LDA discriminant directions, topped up to n_out with the leading
principal components of the LDA-residual, every output z-scored on train. Selected over plain
PCA on validation CKTA for every feature map on both datasets. Deterministic (no seed), so all
seeds of a run share one projection; cached next to the teacher checkpoint it was fitted from.
"""
import os

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def fit_projection(kind, Xtr, ytr, n_out):
    """Returns (W, b) with z = X @ W.T + b train-standardized per output dimension."""
    mu = Xtr.mean(0)
    Xc = (Xtr - mu).numpy().astype(np.float64)
    if kind == "pca":
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Vt[:n_out]
    elif kind == "lda_pca":
        n_cls = int(ytr.max()) + 1
        lda = LinearDiscriminantAnalysis(n_components=n_cls - 1, solver="eigen", shrinkage="auto")
        lda.fit(Xc, ytr.numpy())
        P_lda = lda.scalings_[:, : n_cls - 1].T
        P_lda = P_lda / np.linalg.norm(P_lda, axis=1, keepdims=True)
        resid = Xc - (Xc @ np.linalg.pinv(P_lda)) @ P_lda
        _, _, Vt = np.linalg.svd(resid, full_matrices=False)
        P = np.concatenate([P_lda, Vt[: n_out - (n_cls - 1)]])
    else:
        raise ValueError(f"unknown teacher_projection {kind!r}")
    P = torch.tensor(P, dtype=torch.float32)
    z = (Xtr - mu) @ P.T
    s = z.std(0).clamp(min=1e-8)
    return P / s[:, None], -(mu @ P.T) / s


def load_or_fit(kind, teacher, clean_train_loader, cache_path, n_out, device):
    """Fit once from the frozen teacher's features on unaugmented training images; reuse after."""
    if os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu")
        if blob["kind"] == kind and blob["W"].shape[0] == n_out:
            return blob["W"], blob["b"]
    feats, labels = [], []
    teacher.eval()
    with torch.no_grad():
        for x, y in clean_train_loader:
            feats.append(teacher(x.to(device), return_feats=True)[1][-1].cpu())
            labels.append(y)
    W, b = fit_projection(kind, torch.cat(feats), torch.cat(labels), n_out)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"kind": kind, "W": W, "b": b, "n_train": int(sum(len(y) for y in labels))}, cache_path)
    return W, b
