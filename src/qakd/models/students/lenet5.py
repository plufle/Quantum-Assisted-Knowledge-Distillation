import torch.nn as nn

from qakd.models.registry import register_student


@register_student("lenet5")
class LeNet5(nn.Module):
    """LeNet-5 + AdaptiveAvgPool before FC (~0.07M params) — capacity floor, fast dev loop."""

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, return_feats=False):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        penultimate = self.pool(f4).flatten(1)
        logits = self.fc(penultimate)
        if return_feats:
            return logits, [f1, f2, f3, f4, penultimate]
        return logits
