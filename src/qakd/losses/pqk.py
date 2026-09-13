import torch
import torch.nn as nn


class KernelProjectionHead(nn.Module):
    """Projects penultimate features into an `n_qubits`-dim space for kernel
    construction (CLAUDE.md Loss). Trained jointly with the student/teacher-side
    projection, discarded at export along with everything quantum-related (Rule #1)."""

    def __init__(self, in_dim, n_qubits):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_qubits)

    def forward(self, feats):
        return torch.sigmoid(self.linear(feats)) * torch.pi  # angles in [0, pi]


def normalized_gram(gram):
    """Trace-normalizes a Gram matrix so the Frobenius comparison reflects structure,
    not absolute scale (pqk's quantum kernel and rbf_control's RBF kernel have very
    different raw magnitudes otherwise)."""
    trace = torch.diagonal(gram).sum().clamp(min=1e-12)
    return gram / trace


def kernel_alignment_loss(teacher_gram, student_gram):
    """||K~_t - K~_s||_F^2 (CLAUDE.md Loss)."""
    return (normalized_gram(teacher_gram) - normalized_gram(student_gram)).pow(2).sum()


def gamma_ramp(epoch, ramp_epochs, gamma_max):
    """Linearly ramps gamma 0 -> gamma_max over `ramp_epochs` (CLAUDE.md Loss) so the
    kernel term doesn't dominate CE/KD before the projected features mean anything."""
    if ramp_epochs <= 0:
        return gamma_max
    return gamma_max * min(1.0, (epoch + 1) / ramp_epochs)


def mean_offdiagonal(gram):
    """Mean off-diagonal Gram entry — kernel-concentration check (Trap #3): below 0.05
    the kernel loss is vacuous (every pair looks alike or every pair looks unique)."""
    n = gram.size(0)
    off_diag_sum = gram.sum() - torch.diagonal(gram).sum()
    return off_diag_sum / (n * (n - 1))
