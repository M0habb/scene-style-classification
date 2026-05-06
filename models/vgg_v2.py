import torch
import torch.nn as nn

# VGG16 layer configuration
# integers = out_channels for Conv2d, 'M' = MaxPool2d
CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def make_layers(cfg: list, in_channels: int = 3) -> nn.Sequential:
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [
                nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                nn.BatchNorm2d(v),
                nn.ReLU(inplace=True),
            ]
            in_channels = v
    return nn.Sequential(*layers)


class VGG16_V2(nn.Module):
    def __init__(self, num_classes: int = 17, dropout_rate: float = 0.6):
        super().__init__()

        self.features = make_layers(CFG)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.BatchNorm1d(256),

            nn.Dropout(p=dropout_rate),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.BatchNorm1d(128),

            nn.Dropout(p=dropout_rate),

            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # (N, 512, 7, 7)

        x = self.gap(x)
        x = x.view(x.size(0), -1)  # (N, 512)

        x = self.classifier(x)     # (N, num_classes)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)