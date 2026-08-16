import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, labels, alpha, beta, tau):
    """Hinton logit KD: L = alpha*CE + beta*tau^2*KL(teacher || student) (CLAUDE.md Loss)."""
    ce = F.cross_entropy(student_logits, labels)
    student_log_probs = F.log_softmax(student_logits / tau, dim=1)
    teacher_probs = F.softmax(teacher_logits / tau, dim=1)
    kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (tau ** 2)
    return alpha * ce + beta * kd
