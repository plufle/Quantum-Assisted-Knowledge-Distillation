import torch
import torch.nn.functional as F


def _pairwise_distances(feats):
    diffs = feats.unsqueeze(0) - feats.unsqueeze(1)
    return diffs.pow(2).sum(-1).clamp(min=1e-12).sqrt()


def _distance_potential(feats):
    dist = _pairwise_distances(feats)
    mean_dist = dist[dist > 0].mean().clamp(min=1e-12)
    return dist / mean_dist


def _angle_potential(feats):
    diffs = feats.unsqueeze(0) - feats.unsqueeze(1)  # [N, N, D]: diffs[j, i] = f_i - f_j
    normed = F.normalize(diffs, p=2, dim=-1)
    # angle[j, i, k] = cos angle at vertex j between edges (j->i) and (j->k)
    return torch.einsum("jid,jkd->jik", normed, normed)


def rkd_loss(
    student_feats,
    teacher_feats,
    student_logits,
    labels,
    ce_weight,
    distance_weight,
    angle_weight,
    ce_class_weights=None,
    ce_label_smoothing=0.0,
):
    """Relational KD (Park et al. 2019): distance-wise + angle-wise potentials between
    penultimate embeddings, on top of the hard-label CE (CLAUDE.md: rkd is the direct
    classical ancestor of pqk). Weights default to the paper's 25/50 (see method config)."""
    ce = F.cross_entropy(
        student_logits,
        labels,
        weight=ce_class_weights,
        label_smoothing=ce_label_smoothing,
    )

    teacher_feats = teacher_feats.detach()
    d_loss = F.smooth_l1_loss(_distance_potential(student_feats), _distance_potential(teacher_feats))
    a_loss = F.smooth_l1_loss(_angle_potential(student_feats), _angle_potential(teacher_feats))

    return ce_weight * ce + distance_weight * d_loss + angle_weight * a_loss
