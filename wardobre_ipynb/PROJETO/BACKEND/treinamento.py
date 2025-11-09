# modelo_nn.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dados import preparar_dados_combinados

# --- 1. Definição do Modelo de Rede Neural ---

class RecomendadorContextualNN(nn.Module):
    def __init__(self, n_entradas, n_saidas):
        super(RecomendadorContextualNN, self).__init__()
        
        # Recebe N_ENTRADAS_COMBINADAS (Imagem + Clima)
        self.camada1 = nn.Linear(n_entradas, 128)
        self.camada2 = nn.Linear(128, 64)
        self.camada_saida = nn.Linear(64, n_saidas)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.camada1(x)
        x = self.relu(x)
        x = self.camada2(x)
        x = self.relu(x)
        x = self.camada_saida(x)
        return x

# --- 2. Função de Treinamento ---

def treinar_modelo(X, Y, N_ENTRADAS, N_SAIDAS, N_EPOCAS=50, TAXA_APRENDIZADO=0.001):
    """Inicializa e treina o modelo."""
    
    print("\n--- 3. Configuração e Inicialização do Modelo ---")
    modelo = RecomendadorContextualNN(N_ENTRADAS, N_SAIDAS)
    funcao_perda = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=TAXA_APRENDIZADO)

    print(f"Iniciando treinamento com {N_EPOCAS} épocas...")

    for epoca in range(N_EPOCAS):
        # Forward Pass
        saidas = modelo(X)
        
        # Cálculo da Perda
        perda = funcao_perda(saidas, Y)

        # Backward Pass e Otimização
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()

        if (epoca + 1) % 10 == 0:
            # Cálculo da acurácia de classificação
            _, predito = torch.max(saidas.data, 1)
            acuracia = (predito == Y).sum().item() / Y.size(0)
            print(f'Época [{epoca+1}/{N_EPOCAS}], Perda: {perda.item():.4f}, Acurácia: {acuracia:.4f}')

    print("✅ Treinamento concluído!")
    return modelo

# --- Execução Principal ---

if __name__ == '__main__':
    # 1. Preparação dos dados
    X_COMBINADO, Y, N_ENTRADAS, N_SAIDAS = preparar_dados_combinados()
    
    # 2. Treinamento
    modelo_final = treinar_modelo(X_COMBINADO, Y, N_ENTRADAS, N_SAIDAS)