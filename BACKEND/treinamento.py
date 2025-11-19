
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from dados import preparar_datasets


class ModeloVisaoClima(nn.Module):
    def __init__(self, num_classes, clima_dim):
        super().__init__()
        
        # modelo pré-treinado (remove camada final)
        self.backbone = resnet18(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Identity()
        
        # nova cabeça combinando visão + clima
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


def treinar(local="Curitiba", epochs=10):
    train_ds, test_ds, num_classes = preparar_datasets(local)

    clima_dim = len(train_ds.clima)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    modelo = ModeloVisaoClima(num_classes, clima_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(device)

    ot = optim.Adam(modelo.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    print("\n🚀 Iniciando treino…")

    for ep in range(epochs):
        modelo.train()
        total_loss = 0

        for img, clima, label in train_loader:
            img, clima, label = img.to(device), clima.to(device), label.to(device)

            ot.zero_grad()
            out = modelo(img, clima)
            loss = loss_fn(out, label)
            loss.backward()
            ot.step()

            total_loss += loss.item()

        print(f"Época {ep+1}/{epochs} - Loss: {total_loss:.4f}")

    # avaliação
    modelo.eval()
    corretos = 0
    total = 0

    with torch.no_grad():
        for img, clima, lab in test_loader:
            img, clima, lab = img.to(device), clima.to(device), lab.to(device)
            out = modelo(img, clima)
            _, pred = torch.max(out, 1)
            corretos += (pred == lab).sum().item()
            total += lab.size(0)

    acc = corretos / total
    print(f"\n🎯 Acurácia real no conjunto de teste: {acc*100:.2f}%")

    return modelo