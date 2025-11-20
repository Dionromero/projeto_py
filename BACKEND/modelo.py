# modelo_b.py - MobileNetV2 leve + cabeça para clima
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class ModeloVisaoClimaLite(nn.Module):
    def __init__(self, num_classes, clima_dim, freeze_backbone=True):
        super().__init__()
        self.backbone = mobilenet_v2(weights="IMAGENET1K_V1").features  # pega só features
        # mobilenet_v2.features outputs [N, 1280, 4, 4] typical; usar adaptive pool
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        feat_dim = 1280
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim + clima_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_clima):
        x = self.backbone(x_img)
        x = self.pool(x)   # [N, C, 1, 1]
        x = torch.flatten(x, 1)  # [N, C]
        x = torch.cat([x, x_clima], dim=1)
        return self.head(x)
