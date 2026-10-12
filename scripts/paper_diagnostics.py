"""Heavy, re-runnable diagnostics behind two paper figures/tables. Writes JSON to paper/generated/.

    python scripts/paper_diagnostics.py [--skip-gradvar] [--skip-ckta]

1. gradient_variance.json  — barren-plateau check (Trap #7): variance of d<Z_0>/d(angle_0) over random
   input angles, n_qubits in {2,...,12}, depth 4. Same estimator as scripts/gradient_variance.py.
2. ckta.json               — centered kernel-target alignment (CKTA) of the *teacher* kernel the student is
   asked to match, for the fixed random projection and the train-fitted LDA+PCA projection, measured on
   (a) the validation split (what scripts/teacher_kernel_diagnostics.py reports) and
   (b) random 128-image *training* batches — where the loss is actually computed. The gap between (a)
   and (b) is the label-saturation mechanism behind the fitted-projection result: the teacher was
   fine-tuned and LDA fitted on the training split, so on training batches the fitted target is almost
   the label kernel. Features are the clean training images' teacher penultimate activations, not the
   augmented batches the trainer sees, so (b) is an approximation of the in-training value.
   Test split never read.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
OUT = os.path.join(REPO, "paper", "generated")


def gradvar(qubits=(2, 4, 6, 8, 10, 12), depth=4, samples=200):
    import gradient_variance as gv
    torch.manual_seed(0)
    res = {}
    for n in qubits:
        g = gv._gradient_samples(n, depth, samples, "lightning.qubit")
        res[str(n)] = {"variance": g.var().item(), "mean": g.mean().item(), "n_samples": samples, "depth": depth}
        print(f"  gradvar n_qubits={n:>2}: var {res[str(n)]['variance']:.3e}", flush=True)
    return res


def ckta(n_batches=50, batch=128):
    import teacher_kernel_diagnostics as td
    from hydra import compose, initialize_config_dir
    from qakd.losses.pqk import KernelProjectionHead
    from qakd.losses.teacher_projection import fit_projection
    from qakd.quantum.classical_ctrl import rbf_gram_matrix

    def score(feats_by_map, y):
        L = torch.nn.functional.one_hot(y).float()
        L = L @ L.T
        return {k: td.centered_alignment(rbf_gram_matrix(v, None), L) for k, v in feats_by_map.items()}

    out = {}
    for ds in ("trashnet", "rps_25"):
        with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base=None):
            cfg = compose(config_name="config", overrides=[f"dataset={ds}"])
        feats = td.teacher_features(cfg, ds)
        Xtr, ytr = feats["train"]
        Xva, yva = feats["val"]
        g = torch.Generator().manual_seed(0)
        batches = [torch.randperm(len(ytr), generator=g)[: min(batch, len(ytr))] for _ in range(n_batches)]
        res = {}

        def add(target, per_split):  # per_split: {"val": [scores...], "train_batch": [scores...]}
            res[target] = {s: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                           for s, v in per_split.items()}

        # raw teacher RBF (upper reference)
        L = lambda y: (lambda o: o @ o.T)(torch.nn.functional.one_hot(y).float())  # noqa: E731
        add("raw teacher 2048-d", {
            "val": [td.centered_alignment(rbf_gram_matrix(Xva, None), L(yva))],
            "train_batch": [td.centered_alignment(rbf_gram_matrix(Xtr[b], None), L(ytr[b])) for b in batches]})
        with torch.no_grad():
            # random projection: 5 draws; per draw the train-batch score is the mean over batches
            per = {}
            for d in range(td.N_RANDOM_DRAWS):
                torch.manual_seed(d)
                head = KernelProjectionHead(Xtr.shape[1], 8)
                v = score(td.feature_maps(head(Xva)), yva)
                tb = [score(td.feature_maps(head(Xtr[b])), ytr[b]) for b in batches]
                for k in v:
                    per.setdefault(k, {"val": [], "train_batch": []})
                    per[k]["val"].append(v[k])
                    per[k]["train_batch"].append(float(np.mean([t[k] for t in tb])))
            for k, sp in per.items():
                add(f"random | {k}", sp)
            for kind in ("pca", "lda_pca"):
                W, b_ = fit_projection(kind, Xtr, ytr, 8)
                proj = lambda X: torch.sigmoid(X @ W.T + b_) * torch.pi  # noqa: E731
                v = score(td.feature_maps(proj(Xva)), yva)
                tb = [score(td.feature_maps(proj(Xtr[b])), ytr[b]) for b in batches]
                for k in v:
                    add(f"{kind} | {k}", {"val": [v[k]], "train_batch": [float(np.mean([t[k] for t in tb]))]})
        out[ds] = {"n_train": int(len(ytr)), "n_val": int(len(yva)), "batch": min(batch, len(ytr)),
                   "n_batches": n_batches, "targets": res}
        print(f"  ckta {ds} done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gradvar", action="store_true")
    ap.add_argument("--skip-ckta", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.skip_gradvar:
        print("barren-plateau check", flush=True)
        with open(os.path.join(OUT, "gradient_variance.json"), "w") as f:
            json.dump(gradvar(), f, indent=1)
    if not args.skip_ckta:
        print("teacher-kernel CKTA", flush=True)
        with open(os.path.join(OUT, "ckta.json"), "w") as f:
            json.dump(ckta(), f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
