"""Barren-plateau check (Trap #7): don't exceed 8 qubits without confirming gradient
variance doesn't vanish first. Sweeps n_qubits, sampling many random input angles and
measuring the variance of d<Z_0>/d(angle_0) across them — a McClean et al. (2018)-style
diagnostic. Exponentially decaying variance with n_qubits is the barren-plateau signature.

Usage: python scripts/gradient_variance.py --qubits 2,4,6,8,10,12
"""
import argparse

import torch

from qakd.quantum.kernels import bloch_vectors


def _gradient_samples(n_qubits, depth, n_samples, device_backend):
    grads = []
    for _ in range(n_samples):
        angles = (torch.rand(1, n_qubits) * torch.pi).requires_grad_(True)
        features = bloch_vectors(angles, n_qubits, depth, device_backend)
        z0 = features[:, 2]  # <Z_0> — representative single observable, per McClean et al.
        grad = torch.autograd.grad(z0.sum(), angles)[0]
        grads.append(grad[0, 0].item())
    return torch.tensor(grads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=str, default="2,4,6,8,10,12")
    parser.add_argument("--depth", type=int, default=4, help="matches pqk.yaml's depth")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--device-backend", type=str, default="lightning.qubit")
    args = parser.parse_args()

    qubit_counts = [int(q) for q in args.qubits.split(",")]
    print(f"{'n_qubits':>8} {'grad_variance':>16} {'grad_mean':>14}")
    for n_qubits in qubit_counts:
        grads = _gradient_samples(n_qubits, args.depth, args.samples, args.device_backend)
        variance = grads.var().item()
        mean = grads.mean().item()
        flag = "  <-- possible barren plateau (var < 1e-6)" if variance < 1e-6 else ""
        print(f"{n_qubits:>8} {variance:>16.3e} {mean:>14.3e}{flag}")


if __name__ == "__main__":
    main()
