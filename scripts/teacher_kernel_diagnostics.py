"""How much class structure survives in the *teacher* kernel the student is asked to match?

The kernel-alignment loss can only transfer what the teacher-side Gram matrix contains. The
trainer builds that matrix as: frozen teacher penultimate (2048-d) -> projection -> 8 angles
-> feature map -> RBF (median heuristic). This script measures class separation of that
target at each stage, for the current *fixed random* projection and for projections *fitted on
the training split only* (frozen afterwards), so the projection can be selected on validation.

Selection protocol: fit on train, score on validation. The test split is never read.

Metric: centered kernel-target alignment (CKTA, Cortes et al. 2012) between the kernel and
the ideal label kernel Y Y^T — 1 means the kernel encodes the class partition perfectly.
Also reported: CKA between the PQK kernel and its entanglement-free twin, i.e. how much the
CNOTs change kernel geometry.

    python scripts/teacher_kernel_diagnostics.py --dataset trashnet
"""
import argparse
import os
import sys

import numpy as np
import torch
from hydra import compose, initialize_config_dir

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import qakd.models.teachers  # noqa: E402,F401
from qakd.data.datasets import build_bn_calibration_loader, build_rps25_loaders, build_trashnet_loaders  # noqa: E402
from qakd.losses.pqk import KernelProjectionHead  # noqa: E402
from qakd.losses.teacher_projection import fit_projection  # noqa: E402
from qakd.models.registry import build_teacher  # noqa: E402
from qakd.quantum.classical_ctrl import rbf_gram_matrix  # noqa: E402
from qakd.quantum.kernels import bloch_vectors  # noqa: E402

LOADERS = {"trashnet": build_trashnet_loaders, "rps_25": build_rps25_loaders}
N_RANDOM_DRAWS = 5


def centered_alignment(K, L):
    n = K.shape[0]
    H = torch.eye(n) - 1.0 / n
    Kc, Lc = H @ K @ H, H @ L @ H
    return float((Kc * Lc).sum() / (Kc.norm() * Lc.norm()))


def teacher_features(cfg, dataset):
    cache = os.path.join(REPO_ROOT, "exports", f"_teacher_feats__{dataset}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    train_loader, val_loader, _ = LOADERS[dataset](cfg.dataset)
    teacher = build_teacher(cfg.teacher.name, num_classes=cfg.dataset.num_classes, pretrained=False)
    teacher.load_state_dict(torch.load(os.path.join(
        REPO_ROOT, "checkpoints", "teachers", f"{cfg.teacher.name}__{dataset}.pt"), map_location="cpu"))
    teacher.eval()
    out = {}
    with torch.no_grad():
        for split, loader in [("train", build_bn_calibration_loader(cfg.dataset, train_loader)), ("val", val_loader)]:
            f, y = [], []
            for x, lab in loader:
                f.append(teacher(x, return_feats=True)[1][-1])
                y.append(lab)
            out[split] = (torch.cat(f), torch.cat(y))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(out, cache)
    return out


def feature_maps(angles):
    return {
        "angles (rbf_control)": angles,
        "pqk 16 (RY+CNOT)": bloch_vectors(angles, 8, 4)[:, [i for i in range(24) if i % 3 != 1]],
        "no-CNOT 16 [sin4t,cos4t]": torch.cat([torch.sin(4 * angles), torch.cos(4 * angles)], 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    args = ap.parse_args()
    with initialize_config_dir(config_dir=os.path.join(REPO_ROOT, "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"dataset={args.dataset}"])
    feats = teacher_features(cfg, args.dataset)
    Xtr, ytr = feats["train"]
    Xva, yva = feats["val"]
    L = torch.nn.functional.one_hot(yva).float()
    L = L @ L.T
    print(f"{args.dataset}: train {len(ytr)}, val {len(yva)} (projections fit on train, scored on val)")
    print(f"  raw teacher 2048-d RBF:  CKTA {centered_alignment(rbf_gram_matrix(Xva, None), L):.4f}  (upper reference)")

    with torch.no_grad():
        rows = []
        for d in range(N_RANDOM_DRAWS):
            torch.manual_seed(d)
            head = KernelProjectionHead(Xtr.shape[1], 8)
            rows.append({k: centered_alignment(rbf_gram_matrix(v, None), L)
                         for k, v in feature_maps(head(Xva)).items()})
        print(f"  random projection (mean of {N_RANDOM_DRAWS} draws):")
        for k in rows[0]:
            v = [r[k] for r in rows]
            print(f"     {k:<26} CKTA {np.mean(v):.4f} +/- {np.std(v):.4f}")
        for kind in ["pca", "lda_pca"]:
            W, b = fit_projection(kind, Xtr, ytr, 8)
            ang = torch.sigmoid(Xva @ W.T + b) * torch.pi
            fm = feature_maps(ang)
            print(f"  fitted projection: {kind}")
            for k, v in fm.items():
                print(f"     {k:<26} CKTA {centered_alignment(rbf_gram_matrix(v, None), L):.4f}")
            kq = rbf_gram_matrix(fm["pqk 16 (RY+CNOT)"], None)
            kc = rbf_gram_matrix(fm["no-CNOT 16 [sin4t,cos4t]"], None)
            print(f"     CKA(pqk kernel, no-CNOT kernel) = {centered_alignment(kq, kc):.4f}")


if __name__ == "__main__":
    main()
