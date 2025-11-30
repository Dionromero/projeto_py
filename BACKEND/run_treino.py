
from treinamento import treinar
import torch

if __name__ == "__main__":
    modelo = treinar(epochs=5)
    
    # Salva o modelo final
    torch.save(modelo.state_dict(), "modelo_final_b.pth")
    print("Modelo salvo com sucesso!")