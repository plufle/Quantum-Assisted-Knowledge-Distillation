import torch.nn as nn
import torchvision.models as tvm

from qakd.models.registry import register_teacher


@register_teacher("resnet50_in1k")
class ResNet50Teacher(nn.Module):
    """ResNet-50, ImageNet-pretrained, fine-tuned per dataset then frozen (Trap #6)."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x, return_feats=False):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        penultimate = self.avgpool(f4).flatten(1)
        logits = self.fc(penultimate)
        if return_feats:
            return logits, [f1, f2, f3, f4, penultimate]
        return logits
