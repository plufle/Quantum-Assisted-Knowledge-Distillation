import torch.nn as nn
import torchvision.models as tvm

from qakd.models.common import forward_with_stage_feats
from qakd.models.registry import register_student


@register_student("mobilenetv2_050")
@register_student("mobilenetv2_060")
@register_student("mobilenetv2_035")
class MobileNetV2Student(nn.Module):
    """MobileNetV2, width set by config. x0.35 (~0.4M) is the primary edge target.

    x0.50 (~0.70M) and x0.60 (~0.94M) exist to separate capacity from architecture: the
    only two students where the kernel methods were compared differ in width *and* family
    *and* activation *and* SE blocks *and* the Trap #10 BN flag, so "pqk does better on
    smaller models" could not be told apart from "pqk does better on MobileNetV2". x0.60
    lands within 1.4% of mobilenetv3_small's 930,470 params, making it a capacity-matched
    control for that comparison; x0.50 fills the gap between 0.35 and 0.60."""

    def __init__(self, num_classes, width_mult=0.35):
        super().__init__()
        backbone = tvm.mobilenet_v2(width_mult=width_mult)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(backbone.classifier[1].in_features, num_classes)

    def forward(self, x, return_feats=False):
        h, feats = forward_with_stage_feats(self.features, x, n_stages=4)
        penultimate = self.avgpool(h).flatten(1)
        logits = self.fc(penultimate)
        if return_feats:
            return logits, feats + [penultimate]
        return logits
