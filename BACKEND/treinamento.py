
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dados import preparar_datasets
from modelo import ModeloVisaoClimaLite

def treinar(local="Curitiba", epochs=5, batch_size=64, lr=1e-3, freeze_backbone=True):
    train_ds, test_ds, num_classes = preparar_datasets(local)
    clima_dim = len(train_ds.clima)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    modelo = ModeloVisaoClimaLite(num_classes, clima_dim, freeze_backbone=freeze_backbone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(device)

    criterio = nn.CrossEntropyLoss()
    # só otimizar parâmetros com requires_grad=True
    optim_params = [p for p in modelo.parameters() if p.requires_grad]
    otimizador = optim.Adam(optim_params, lr=lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    torch.backends.cudnn.benchmark = True

    print("\n🚀 Iniciando treino (forma B - leve)... device:", device)

    for ep in range(epochs):
        modelo.train()
        total_loss = 0.0
        for imgs, clima, labels in train_loader:
            imgs = imgs.to(device)
            clima = clima.to(device)
            labels = labels.to(device)

            otimizador.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                outputs = modelo(imgs, clima)
                loss = criterio(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(otimizador)
            scaler.update()

            total_loss += loss.item()

        print(f"Época {ep+1}/{epochs} - Loss: {total_loss:.4f}")

    # avaliação
    modelo.eval()
    corretos = 0
    total = 0
    with torch.no_grad():
        for imgs, clima, labels in test_loader:
            imgs = imgs.to(device)
            clima = clima.to(device)
            labels = labels.to(device)
            outputs = modelo(imgs, clima)
            _, preds = torch.max(outputs, 1)
            corretos += (preds == labels).sum().item()
            total += labels.size(0)

    acc = corretos / total if total > 0 else 0.0
    print(f"\n Acurácia no teste: {acc*100:.2f}%")
    return modelo
