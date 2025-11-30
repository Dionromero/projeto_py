import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dados import preparar_datasets
from modelo import ModeloVisaoClimaLite

def treinar(epochs=10, batch_size=64, lr=1e-4, freeze_backbone=False):
    
    train_ds, test_ds, num_classes = preparar_datasets()
    
    # Dataloaders
    # Nota: Se der erro de "BrokenPipe" no Windows, mude num_workers para 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Instancia modelo sem dimensão de clima
    modelo = ModeloVisaoClimaLite(num_classes, freeze_backbone=freeze_backbone)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(device)

    criterio = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=lr)
    
    # --- CORREÇÃO AQUI: Removido 'verbose=True' ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode='min', factor=0.5, patience=2)
    # ----------------------------------------------
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    torch.backends.cudnn.benchmark = True

    print(f"\n Iniciando treino (Visão Pura)... Device: {device}")

    for ep in range(epochs):
        modelo.train()
        total_loss = 0.0
        
        for imgs, labels, _ in train_loader: 
            imgs = imgs.to(device)
            labels = labels.to(device)

            otimizador.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                # Modelo recebe só a imagem
                outputs = modelo(imgs)
                loss = criterio(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(otimizador)
            scaler.update()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Pega o Learning Rate atual para imprimir (já que removemos o verbose)
        lr_atual = otimizador.param_groups[0]['lr']
        print(f"Época {ep+1}/{epochs} - Loss Médio: {avg_loss:.4f} - LR: {lr_atual:.6f}")
        
        # Atualiza o scheduler
        scheduler.step(avg_loss)

    # Avaliação
    modelo.eval()
    corretos = 0
    total = 0
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = modelo(imgs)
            _, preds = torch.max(outputs, 1)
            corretos += (preds == labels).sum().item()
            total += labels.size(0)

    acc = corretos / total if total > 0 else 0.0
    print(f"\n Acurácia no teste: {acc*100:.2f}%")
    return modelo