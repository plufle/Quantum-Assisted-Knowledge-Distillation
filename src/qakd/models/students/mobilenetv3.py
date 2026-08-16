import torch.nn as nn
import torchvision.models as tvm

from qakd.models.common import forward_with_stage_feats
from qakd.models.registry import register_student


@register_student("mobilenetv3_small")
class MobileNetV3SmallStudent(nn.Module):
    """MobileNetV3-Small (~1.0M params) — primary edge target."""

    def __init__(self, num_classes):
        super().__init__()
        backbone = tvm.mobilenet_v3_small()
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(backbone.classifier[0].in_features, num_classes)

    def forward(self, x, return_feats=False):
        h, feats = forward_with_stage_feats(self.features, x, n_stages=4)
        penultimate = self.avgpool(h).flatten(1)
        logits = self.fc(penultimate)
        if return_feats:
            return logits, feats + [penultimate]
        return logits
