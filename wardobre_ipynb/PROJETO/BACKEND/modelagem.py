import torch
import torch.nn as nn
import torch.optim as optim
import clima as clima

# Modelo simples
class ModeloRoupas(nn.Module):
    def __init__(self):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.rede(x)

modelo = ModeloRoupas()

# Dados fictícios (temperatura, umidade, chuva)
# e as "roupas" correspondentes (0=casaco, 1=camisa, 2=regata, 3=jaqueta)
entradas = torch.tensor([
    [5, 90, 1],   # frio e chuva
    [10, 70, 0],  # frio leve
    [25, 50, 0],  # agradável
    [35, 30, 0],  # calor
], dtype=torch.float32)

rotulos = torch.tensor([0, 3, 1, 2])  # roupas ideais

# Treinar o modelo
criterio = nn.CrossEntropyLoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.01)

for epoca in range(500):
    saida = modelo(entradas)
    perda = criterio(saida, rotulos)
    otimizador.zero_grad()
    perda.backward()
    otimizador.step()

print("Treinamento concluído ✅")


