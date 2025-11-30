import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class ModeloVisaoClimaLite(nn.Module):
    # Note que removemos 'clima_dim' dos argumentos aqui embaixo
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        # Carrega a MobileNetV2
        self.backbone = mobilenet_v2(weights="IMAGENET1K_V1").features 
        
        # MobileNetV2 sai com 1280 canais
        feat_dim = 1280
        
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Pooling para garantir tamanho 1x1 antes do Flatten
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256), # Antes era feat_dim + clima_dim. Agora é só feat_dim.
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    # Note que removemos 'x_clima' daqui também
    def forward(self, x_img): 
        x = self.backbone(x_img)      # [N, 1280, H, W]
        x = self.pool(x)              # [N, 1280, 1, 1]
        x = torch.flatten(x, 1)       # [N, 1280]
        
        return self.head(x)