import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, labels, alpha, beta, tau, ce_class_weights=None, ce_label_smoothing=0.0):
    """Hinton logit KD: L = alpha*CE + beta*tau^2*KL(teacher || student) (CLAUDE.md Loss)."""
    ce = F.cross_entropy(
        student_logits,
        labels,
        weight=ce_class_weights,
        label_smoothing=ce_label_smoothing,
    )
    student_log_probs = F.log_softmax(student_logits / tau, dim=1)
    teacher_probs = F.softmax(teacher_logits / tau, dim=1)
    kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (tau ** 2)
    return alpha * ce + beta * kd
