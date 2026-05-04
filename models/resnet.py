import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    """
    ResNet-50 teacher model built on torchvision pretrained weights.
    The final FC layer is replaced to match your num_classes.
    All backbone layers are unfrozen — fine-tune the whole network.
    """
    def __init__(self, num_classes: int = 17, freeze_backbone: bool = False):
        super().__init__()

        # Load pretrained ResNet-50
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Replace the final FC (1000 ImageNet classes → your num_classes)
        in_features = backbone.fc.in_features       # 2048 for ResNet-50
        backbone.fc = nn.Sequential( # type: ignore
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)