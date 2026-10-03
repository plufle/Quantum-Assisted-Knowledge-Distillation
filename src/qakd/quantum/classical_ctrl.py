import torch


def rbf_gram_matrix(x, bandwidth=None):
    """Batch RBF Gram matrix: K_ij = exp(-||x_i - x_j||^2 / (2*bandwidth^2)) — rbf_control's
    classical twin to pqk's quantum kernel (CLAUDE.md Rule #2), built at the same projected
    dimension (`n_qubits`) as pqk for a matched comparison."""
    dists_sq = torch.cdist(x, x, p=2).pow(2)
    if bandwidth is None:
        bandwidth = _median_heuristic_bandwidth(dists_sq)
    return torch.exp(-dists_sq / (2 * bandwidth ** 2))


def entanglement_free_features(angles, depth):
    """[sin(depth*a), cos(depth*a)] per angle — exactly the PQK circuit with its CNOTs deleted.

    The PQK feature map is `depth` repeats of RY(a_q) + a CNOT ring, read out as <X_q>, <Z_q>
    (<Y_q> is identically zero for a real-amplitude circuit). Remove the CNOTs and each qubit
    sees RY(a_q)^depth = RY(depth*a_q) on |0>, so <X_q> = sin(depth*a_q), <Z_q> = cos(depth*a_q)
    (verified against the simulator to 6e-8). Same 2n active coordinates, same [-1, 1] range,
    no trainable parameters — so a kernel on this map differs from PQK *only* by entanglement,
    and is the tightest matched classical control available."""
    return torch.cat([torch.sin(depth * angles), torch.cos(depth * angles)], dim=1)


_RFF_CACHE = {}


def spectrum_rff_features(angles, width, max_freq, seed):
    """Random Fourier features on the PQK circuit's own frequency spectrum:
    sqrt(2/width) * cos(angles @ W.T + b), W integer-uniform in [-max_freq, max_freq], b ~ U[0, 2pi).

    A depth-L data-re-uploading circuit with Pauli rotations outputs trigonometric polynomials
    whose frequencies are the integers {-L..L} in each input angle (Schuld, Sweke & Meyer,
    2021). With max_freq = depth this draws *random* members of exactly the function class the
    circuit implements — same frequencies, features mixing all angles as entanglement does,
    same width — so it is a generic classical control with no tuned scale. W and b come from a
    private generator with a fixed seed: identical across training seeds (like the fixed
    circuit) and they never touch the global RNG, so pairing with other arms is preserved."""
    n = angles.shape[1]
    key = (n, width, max_freq, seed, angles.device, angles.dtype)
    if key not in _RFF_CACHE:
        g = torch.Generator().manual_seed(seed)
        W = torch.randint(-max_freq, max_freq + 1, (width, n), generator=g).to(angles.dtype)
        b = torch.rand(width, generator=g) * 2 * torch.pi
        _RFF_CACHE[key] = (W.to(angles.device), b.to(angles.device, angles.dtype))
    W, b = _RFF_CACHE[key]
    return (2.0 / width) ** 0.5 * torch.cos(angles @ W.T + b)


def _median_heuristic_bandwidth(dists_sq):
    """Median-heuristic bandwidth, used whenever `method.bandwidth` is null in config
    (tune with the same budget given to the quantum branch — CLAUDE.md Methods)."""
    n = dists_sq.size(0)
    off_diag = dists_sq[~torch.eye(n, dtype=torch.bool, device=dists_sq.device)]
    median_sq = off_diag.median().clamp(min=1e-12)
    return median_sq.sqrt()
