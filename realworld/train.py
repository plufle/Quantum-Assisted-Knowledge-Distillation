"""Train the real-world RPS teacher and students (separate from the QAKD research runs).

    python realworld/train.py --role teacher
    python realworld/train.py --role student --method kd
    python realworld/train.py --role student --method pqk_adaptive

Teacher: ResNet-50, ImageNet-pretrained (the research teacher class, 4 outputs).
Students: MobileNetV3-Small, ImageNet-pretrained, distilled from the teacher with exactly the
research losses, imported read-only from src/qakd:
  kd            alpha*CE + beta*tau^2*KL, tau=1 (the research's rps_25 KD setting)
  pqk_adaptive  CE + KL (tau=0.5) + gamma * ||K~_t - K~_s||_F^2 on an 8-qubit, depth-4 projected
                quantum kernel with per-batch median lambda, gamma ramped 0->1 over 10 epochs,
                teacher-side projection fixed at random init (research Trap #13)
Checkpoint selection on real-validation CE loss (research Trap #11: F1 on ~40 images is too
coarse to pick epochs). Outputs: realworld/runs/{name}/ — best.pt, metrics.json, log.txt.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_rps  # noqa: E402
from qakd.losses.kd import kd_loss  # noqa: E402
from qakd.losses.pqk import KernelProjectionHead, gamma_ramp, kernel_alignment_loss, mean_offdiagonal  # noqa: E402
from qakd.models.teachers.resnet import ResNet50Teacher  # noqa: E402

RUNS = os.path.join(data_rps.REPO, "realworld", "runs")
TEACHER_NAME = "teacher_resnet50"
KD = {"alpha": 0.5, "beta": 0.5, "tau": 1.0}
PQK = {"alpha": 0.5, "beta": 0.5, "tau": 0.5, "n_qubits": 8, "depth": 4, "gamma": 1.0, "ramp": 10}


class Student(nn.Module):
    """torchvision MobileNetV3-Small with ImageNet weights; returns pooled 576-d features for the
    kernel term (the research students' penultimate feature for mobilenetv3_small)."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features, self.avgpool, self.classifier = m.features, m.avgpool, m.classifier
        self.classifier[3] = nn.Linear(self.classifier[3].in_features, num_classes)

    def forward(self, x, return_feats=False):
        f = self.avgpool(self.features(x)).flatten(1)
        logits = self.classifier(f)
        return (logits, f) if return_feats else logits


def evaluate(model, loader):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x))
            labels.append(y)
    logits, labels = torch.cat(logits), torch.cat(labels)
    preds = logits.argmax(1).numpy()
    y = labels.numpy()
    present = sorted(set(y.tolist()))
    return {"ce_loss": float(F.cross_entropy(logits, labels)), "accuracy": float((preds == y).mean()),
            "macro_f1": float(f1_score(y, preds, average="macro", labels=present)),
            "confusion": confusion_matrix(y, preds, labels=list(range(len(data_rps.CLASSES)))).tolist(), "n": int(len(y))}


def load_teacher():
    teacher = ResNet50Teacher(num_classes=len(data_rps.CLASSES), pretrained=False)
    teacher.load_state_dict(torch.load(os.path.join(RUNS, TEACHER_NAME, "best.pt"), map_location="cpu"))
    teacher.eval()
    for p in teacher.parameters():  # research Trap #6
        p.requires_grad_(False)
    return teacher


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", required=True, choices=["teacher", "student"])
    ap.add_argument("--method", default="kd", choices=["kd", "pqk_adaptive"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=0, help="debug: stop each epoch after N batches")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    teacher_role = args.role == "teacher"
    epochs = args.epochs or (12 if teacher_role else 25)
    batch = args.batch or (64 if teacher_role else 128)  # >= 64 for the batch Gram matrix (research Trap #2)
    lr = args.lr or (1e-4 if teacher_role else 5e-4)
    name = TEACHER_NAME if teacher_role else f"student_mnv3s_{args.method}_s{args.seed}"
    out = os.path.join(RUNS, name)
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log.txt"), "a")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    loaders, sizes = data_rps.build_loaders(batch, num_workers=args.workers, seed=args.seed)
    n_cls = len(data_rps.CLASSES)
    if teacher_role:
        model, teacher = ResNet50Teacher(num_classes=n_cls, pretrained=True), None
    else:
        model, teacher = Student(n_cls), load_teacher()
    params = list(model.parameters())
    kernel = None
    if not teacher_role and args.method == "pqk_adaptive":
        from qakd.quantum.kernels import pqk_gram_matrix
        student_proj = KernelProjectionHead(576, PQK["n_qubits"])
        torch.manual_seed(1000 + args.seed)
        teacher_proj = KernelProjectionHead(2048, PQK["n_qubits"]).eval()  # fixed random target (Trap #13)
        for p in teacher_proj.parameters():
            p.requires_grad_(False)
        params += list(student_proj.parameters())
        kernel = (student_proj, teacher_proj, pqk_gram_matrix)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    say(f"{name}: epochs {epochs}, batch {batch}, lr {lr}, data {sizes}")

    best, best_epoch, t_start = float("inf"), -1, time.time()
    for epoch in range(epochs):
        model.train()
        if kernel:
            kernel[0].train()
        gamma = gamma_ramp(epoch, PQK["ramp"], PQK["gamma"]) if kernel else 0.0
        t0, total, n, offdiag = time.time(), 0.0, 0, []
        for b, (x, y) in enumerate(loaders["train"]):
            if args.max_batches and b >= args.max_batches:
                break
            if teacher_role:
                loss = F.cross_entropy(model(x), y)
            else:
                logits, s_feats = model(x, return_feats=True)
                with torch.no_grad():
                    t_logits, t_feats = teacher(x, return_feats=True)
                cfg = PQK if kernel else KD
                loss = kd_loss(logits, t_logits, y, cfg["alpha"], cfg["beta"], cfg["tau"])
                if kernel:
                    s_proj, t_proj, gram = kernel
                    k_s = gram(s_proj(s_feats), PQK["n_qubits"], PQK["depth"], None)
                    with torch.no_grad():
                        k_t = gram(t_proj(t_feats[-1]), PQK["n_qubits"], PQK["depth"], None)
                    loss = loss + gamma * kernel_alignment_loss(k_t, k_s)
                    offdiag.append(mean_offdiagonal(k_s).item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        val = evaluate(model, loaders["val"])
        extra = f", gamma {gamma:.2f}, K offdiag {np.mean(offdiag):.2f}" if kernel else ""
        say(f"epoch {epoch + 1}/{epochs}: train loss {total / max(n, 1):.4f}, val CE {val['ce_loss']:.4f}, "
            f"val acc {val['accuracy']:.3f}, val F1 {val['macro_f1']:.3f}{extra} ({time.time() - t0:.0f} s)")
        if val["ce_loss"] < best:
            best, best_epoch = val["ce_loss"], epoch + 1
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location="cpu"))
    metrics = {"name": name, "role": args.role, "method": None if teacher_role else args.method, "seed": args.seed,
               "classes": data_rps.CLASSES, "best_epoch": best_epoch, "epochs": epochs, "batch": batch, "lr": lr,
               "params": sum(p.numel() for p in model.parameters()), "train_minutes": round((time.time() - t_start) / 60, 1),
               "data": sizes, "val": evaluate(model, loaders["val"]), "test_real": evaluate(model, loaders["test"]),
               "test_rps_rendered": evaluate(model, loaders["rps_test"])}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    say(f"best epoch {best_epoch}: real test acc {metrics['test_real']['accuracy']:.3f} "
        f"(F1 {metrics['test_real']['macro_f1']:.3f}), rendered RPS test acc {metrics['test_rps_rendered']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
