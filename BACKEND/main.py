# main_b.py
from treinamento_b import treinar_b
from imagem import predizer_imagem
import torch, os

MODELO_PATH = "modelo_final_b.pth"

def salvar_modelo(modelo, path=MODELO_PATH):
    torch.save(modelo.state_dict(), path)
    print("Modelo salvo em:", path)

if __name__ == "__main__":
    # Treina por exemplo 3 épocas (rápido)
    modelo = treinar_b(epochs=3, batch_size=128, lr=1e-3, freeze_backbone=True)
    salvar_modelo(modelo)
    # predizer exemplo (coloque imagem em imagens/exemplo.jpg)
    # predizer_imagem("imagens/exemplo.jpg", MODELO_PATH)
