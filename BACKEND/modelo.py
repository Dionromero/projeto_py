import torch
import torch.nn as nn
from torchvision.models import resnet18


class ModeloVisaoClima(nn.Module):
    def __init__(self, num_classes, clima_dim):
        super().__init__()
        
        # Load pretrained ResNet18
        self.backbone = resnet18(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Identity()  # remove classification head
        
        # New head combining visual features + clima
        self.fc = nn.Sequential(
            nn.Linear(512 + clima_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, clima):
        feat_img = self.backbone(img)
        combined = torch.cat([feat_img, clima], dim=1)
        return self.fc(combined)
