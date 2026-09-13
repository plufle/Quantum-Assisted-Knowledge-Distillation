import torch


def rbf_gram_matrix(x, bandwidth=None):
    """Batch RBF Gram matrix: K_ij = exp(-||x_i - x_j||^2 / (2*bandwidth^2)) — rbf_control's
    classical twin to pqk's quantum kernel (CLAUDE.md Rule #2), built at the same projected
    dimension (`n_qubits`) as pqk for a matched comparison."""
    dists_sq = torch.cdist(x, x, p=2).pow(2)
    if bandwidth is None:
        bandwidth = _median_heuristic_bandwidth(dists_sq)
    return torch.exp(-dists_sq / (2 * bandwidth ** 2))


def _median_heuristic_bandwidth(dists_sq):
    """Median-heuristic bandwidth, used whenever `method.bandwidth` is null in config
    (tune with the same budget given to the quantum branch — CLAUDE.md Methods)."""
    n = dists_sq.size(0)
    off_diag = dists_sq[~torch.eye(n, dtype=torch.bool, device=dists_sq.device)]
    median_sq = off_diag.median().clamp(min=1e-12)
    return median_sq.sqrt()
