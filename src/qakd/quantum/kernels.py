import pennylane as qml
import torch

# Trap #1 — always lightning.qubit, never default.qubit (50x slower).
_DEVICE_CACHE = {}
_QNODE_CACHE = {}


def _get_device(n_qubits, device_backend):
    key = (n_qubits, device_backend)
    if key not in _DEVICE_CACHE:
        _DEVICE_CACHE[key] = qml.device(device_backend, wires=n_qubits)
    return _DEVICE_CACHE[key]


def _feature_map(angles, n_qubits, depth):
    """Data re-uploading feature map: `depth` repeats of [RY(angle) per qubit, then a
    CNOT entangling ring] (CLAUDE.md pqk: n_qubits=8, depth=4). Fixed and non-trainable
    — only the classical projection layer upstream of this is trained; `depth` controls
    how entangled (and how expressive/potentially barren — Trap #7) the embedding is."""
    # NOTE: RY and CNOT are both real matrices, so starting from |0..0> the state keeps real
    # amplitudes and every <Y_q> is identically zero (verified: max |<Y>| = 0.0). The measured
    # XYZ feature therefore has 2n active coordinates, not 3n (16 at 8 qubits; 44, not 52, with
    # ZZ). Left as-is so frozen results stay bit-reproducible — the zero columns add nothing to
    # any distance or median — but adding a phase gate (e.g. RZ) would be needed to use them.
    for _ in range(depth):
        for i in range(n_qubits):
            qml.RY(angles[:, i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])


def _build_qnode(n_qubits, depth, device_backend, include_zz=False):
    dev = _get_device(n_qubits, device_backend)

    def circuit(angles):
        _feature_map(angles, n_qubits, depth)
        obs = [o(i) for i in range(n_qubits) for o in (qml.PauliX, qml.PauliY, qml.PauliZ)]
        if include_zz:
            # Two-body <Z_i Z_j>. Single-qubit marginals discard exactly the correlations
            # entanglement creates — and the more qubits/depth, the more the reduced states
            # drift toward maximally mixed (measured: mean |r| 0.541 at 4 qubits -> 0.313 at
            # 12), draining the signal a purely single-qubit kernel reads. These n(n-1)/2
            # correlators put that information back while staying polynomial, so the kernel
            # is still classically tractable (the point of a *projected* quantum kernel).
            obs += [qml.PauliZ(i) @ qml.PauliZ(j)
                    for i in range(n_qubits) for j in range(i + 1, n_qubits)]
        return [qml.expval(o) for o in obs]

    return qml.QNode(circuit, dev, interface="torch", diff_method="adjoint")


def bloch_vectors(angles, n_qubits, depth, device_backend="lightning.qubit", include_zz=False):
    """Runs a batch of `n_qubits` angles through the fixed quantum feature map and
    returns each sample's single-qubit reduced-density-matrix Bloch vectors, flattened
    to [batch, 3*n_qubits] (<X_q>, <Y_q>, <Z_q> per qubit) — the classically-tractable
    'projected' feature behind the projected quantum kernel (CLAUDE.md Loss). Computing
    only expectation values (never the full 2^n statevector) is what keeps this
    poly-time classically and lets `diff_method=adjoint` work at all."""
    key = (n_qubits, depth, device_backend, include_zz)
    if key not in _QNODE_CACHE:
        _QNODE_CACHE[key] = _build_qnode(n_qubits, depth, device_backend, include_zz)
    qnode = _QNODE_CACHE[key]
    return torch.stack(qnode(angles), dim=-1)


def pqk_gram_matrix(angles, n_qubits, depth, lambda_, device_backend="lightning.qubit", include_zz=False):
    """Projected quantum kernel Gram matrix (CLAUDE.md Loss):
    K_ij = exp(-lambda * sum_q ||rho_q(x_i) - rho_q(x_j)||_F^2).

    For single-qubit density matrices rho = 1/2(I + r.sigma), sum_q ||rho_q(x_i) -
    rho_q(x_j)||_F^2 reduces to 0.5 * ||bloch(x_i) - bloch(x_j)||^2 (trace(sigma_a
    sigma_b) = 2*delta_ab), so the exponentially-large statevector never needs to be
    formed — the classical Bloch-vector distance is exactly the quantity CLAUDE.md's
    formula calls for.

    `lambda_=None` selects a per-batch median heuristic instead of a fixed value. This
    exists to remove a structural asymmetry with `rbf_control`: the classical twin has
    always recalibrated its bandwidth to each batch (median heuristic), pinning its
    kernel concentration to ~0.60 on every pair, while a fixed lambda let pqk's
    concentration drift over 0.40-0.63 across pairs — i.e. the two "matched" twins were
    not matched on kernel sharpness at all. The scaling below mirrors the classical
    formula exactly (K = exp(-0.5) at the median distance), so both branches sit in the
    same operating range and the comparison isolates the kernel, not its calibration."""
    features = bloch_vectors(angles, n_qubits, depth, device_backend, include_zz)
    dists_sq = 0.5 * torch.cdist(features, features, p=2).pow(2)
    if lambda_ is None:
        lambda_ = _median_heuristic_lambda(dists_sq)
    return torch.exp(-lambda_ * dists_sq)


def _median_heuristic_lambda(dists_sq):
    """lambda = 0.5 / median(off-diagonal d^2), the exact analogue of `rbf_control`'s
    bandwidth = sqrt(median(d^2)) with K = exp(-d^2 / 2*bandwidth^2)."""
    n = dists_sq.size(0)
    off_diag = dists_sq[~torch.eye(n, dtype=torch.bool, device=dists_sq.device)]
    return 0.5 / off_diag.median().clamp(min=1e-12)
