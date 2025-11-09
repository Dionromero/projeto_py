# treinamento.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ascii_magic
import os
from dados import preparar_dados_combinados
from datasets import load_dataset


# --- 1. Definição do Modelo de Rede Neural ---
class RecomendadorContextualNN(nn.Module):
    def __init__(self, n_entradas, n_saidas):
        super(RecomendadorContextualNN, self).__init__()
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


# --- 2. Função para mostrar imagem como arte ASCII ---
def mostrar_imagem_no_terminal(item_dataset, index, classes):
    """Exibe uma imagem do dataset como arte ASCII no terminal."""
    img_pil = item_dataset[index]['image']
    temp_filename = "temp_image_visualizacao.png"
    img_pil.save(temp_filename)

    print("\n" + "=" * 60)
    print(f"👁️  AMOSTRA VISUALIZADA (Índice: {index})")
    print(f"🎽  Classe Original: {classes.get(item_dataset[index]['label'], 'Desconhecida')}")
    print("=" * 60)

    try:
        my_art = ascii_magic.from_image_file(
            temp_filename,
            columns=60,  # tamanho da arte
            char="█"     # caractere usado pra desenhar
        )
        ascii_magic.to_terminal(my_art)
    except Exception as e:
        print(f"❌ Erro ao renderizar arte ASCII: {e}")
    finally:
        os.remove(temp_filename)
        print("=" * 60 + "\n")


# --- 3. Função de Treinamento ---
def treinar_modelo(X, Y, N_ENTRADAS, N_SAIDAS, df_train, N_EPOCAS=30, TAXA_APRENDIZADO=0.001):
    """Treina o modelo e mostra uma amostra de roupa no terminal."""
    modelo = RecomendadorContextualNN(N_ENTRADAS, N_SAIDAS)
    funcao_perda = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=TAXA_APRENDIZADO)

    CLASSES = {
        0: 'T-shirt/Top', 1: 'Calça', 2: 'Pulôver', 3: 'Vestido', 4: 'Casaco',
        5: 'Sandália', 6: 'Camisa', 7: 'Tênis', 8: 'Bolsa', 9: 'Bota', 10: 'Outro'
    }

    print("\n--- INICIANDO TREINAMENTO ---")

    for epoca in range(N_EPOCAS):
        # Mostra uma imagem diferente a cada 10 épocas
        if epoca % 10 == 0:
            index = np.random.randint(0, len(df_train))
            mostrar_imagem_no_terminal(df_train, index=index, classes=CLASSES)

        saidas = modelo(X)
        perda = funcao_perda(saidas, Y)
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()

        if (epoca + 1) % 5 == 0:
            _, predito = torch.max(saidas.data, 1)
            acuracia = (predito == Y).sum().item() / Y.size(0)
            print(f"Época [{epoca+1}/{N_EPOCAS}] - Perda: {perda.item():.4f} - Acurácia: {acuracia:.4f}")

    print("\n✅ Treinamento concluído!")
    return modelo


# --- 4. Execução Principal ---
if __name__ == '__main__':
    print("--- 1. Preparando dados ---")
    X_COMBINADO, Y, N_ENTRADAS, N_SAIDAS = preparar_dados_combinados()

    print("\n--- 2. Carregando dataset ---")
    dataset = load_dataset("samokosik/clothes_simplified")['train']

    print("\n--- 3. Iniciando modelo ---")
    modelo_final = treinar_modelo(
        X_COMBINADO,
        Y,
        N_ENTRADAS,
        N_SAIDAS,
        df_train=dataset,
        N_EPOCAS=30
    )
