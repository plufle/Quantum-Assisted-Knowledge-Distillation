import torch.nn as nn
import torchvision.models as tvm

from qakd.models.registry import register_student


@register_student("shufflenetv2_050")
class ShuffleNetV2Student(nn.Module):
    """ShuffleNetV2 x0.5 (~0.4M params) — cross-family control.

    torchvision only ships fixed named width variants (no arbitrary width_mult),
    so `width_mult` is accepted for config-symmetry with the other students but
    unused here — "050" in the name already pins the variant to x0_5.
    """

    def __init__(self, num_classes, width_mult=None):
        super().__init__()
        backbone = tvm.shufflenet_v2_x0_5()
        self.conv1 = backbone.conv1
        self.maxpool = backbone.maxpool
        self.stage2 = backbone.stage2
        self.stage3 = backbone.stage3
        self.stage4 = backbone.stage4
        self.conv5 = backbone.conv5
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x, return_feats=False):
        h = self.maxpool(self.conv1(x))
        f1 = self.stage2(h)
        f2 = self.stage3(f1)
        f3 = self.stage4(f2)
        f4 = self.conv5(f3)
        penultimate = self.avgpool(f4).flatten(1)
        logits = self.fc(penultimate)
        if return_feats:
            return logits, [f1, f2, f3, f4, penultimate]
        return logits
