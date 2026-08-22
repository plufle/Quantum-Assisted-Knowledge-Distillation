import torch
import torch.nn as nn


def use_batch_stats_only(model):
    """Disables BatchNorm running-stat tracking so normalization always uses the
    current batch's own statistics, in train AND eval mode. Needed for students
    trained from scratch on datasets with too few batches/epoch for running stats
    to stabilize (Trap #10) — otherwise eval-mode BN can degenerate to a frozen,
    input-invariant output even while train-mode (per-batch stats) looks fine."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return model


def calibrate_batch_norm(model, loader, device):
    """Populate deterministic BatchNorm statistics from clean training images.

    In the tiny RPS regime we deliberately train with per-batch statistics, because
    only five optimisation batches per epoch cannot produce reliable running
    statistics.  That setting must *not* be used for validation or test, however:
    otherwise predictions change with the evaluator's batch boundaries.  This
    calibration pass makes one cumulative, no-gradient pass over the unaugmented
    training split before every evaluation.
    """
    batch_norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not batch_norms:
        return

    was_training = model.training
    for module in batch_norms:
        module.track_running_stats = True
        module.running_mean = torch.zeros(module.num_features, device=device)
        module.running_var = torch.ones(module.num_features, device=device)
        module.num_batches_tracked = torch.zeros((), dtype=torch.long, device=device)
        # Cumulative averaging gives each calibration batch equal, reproducible
        # weight and prevents a late mini-batch from dominating the statistics.
        module.momentum = None

    # Keep every non-BN layer (notably dropout) in inference mode.  Each BN
    # module is explicitly put into train mode so it consumes and accumulates
    # the calibration batch statistics.
    model.eval()
    for module in batch_norms:
        module.train()
    with torch.no_grad():
        for images, _ in loader:
            if images.size(0) < 2:
                continue
            model(images.to(device))
    model.train(was_training)


def forward_with_stage_feats(sequential_module, x, n_stages=4):
    """Runs `x` through a `nn.Sequential` feature extractor, snapshotting its output
    at `n_stages` evenly-spaced depth checkpoints (not necessarily semantic stage
    boundaries — just depth checkpoints so every student can satisfy the
    `forward(x, return_feats=True) -> logits, [f1..f4, penult]` interface)."""
    length = len(sequential_module)
    cut_at = sorted({round(length * i / n_stages) for i in range(1, n_stages + 1)})
    cut_at[-1] = length
    while len(cut_at) < n_stages:
        cut_at.insert(0, cut_at[0])

    feats = []
    h = x
    cut_i = 0
    for i, layer in enumerate(sequential_module):
        h = layer(h)
        if cut_i < len(cut_at) and (i + 1) == cut_at[cut_i]:
            feats.append(h)
            cut_i += 1
    return h, feats
