# dataset_preparacao.py

from datasets import load_dataset       
import numpy as np
import torch
from clima import obter_dados_clima, criar_dataframes, processar_dados_horarios 
from imagem import imagem_para_json


# Tamanho fixo para redimensionamento da imagem
IMG_SIZE = (32, 32)
IMG_FLAT_DIM = IMG_SIZE[0] * IMG_SIZE[1] * 3 

def extract_features(item):
    """Redimensiona, normaliza, achata a imagem e gera JSON."""
    try:
        img = item['image'].resize(IMG_SIZE)

        # transforma imagem em JSON
        img_json = imagem_para_json(img)

        # Features numéricas (se você ainda quiser usar)
        features = np.array(img, dtype=np.float32).flatten() / 255.0

        # Agora retorna também o JSON
        return features, item['label'], img_json

    except Exception as e:
        print(f"Erro ao processar item: {e}")
        return None, None, None

def preparar_dados_combinados(local_clima="Curitiba"):
    """
    Carrega o dataset de imagens, busca o clima e combina ambos os tensores.
    """
    print("--- 1. Preparação dos Dados Climáticos ---")
    
    # Execução das suas funções de Clima:
    dados = obter_dados_clima(local_clima)
    df, df_forecast = criar_dataframes(dados)
    VETOR_CLIMA = processar_dados_horarios(df_forecast)
    # -----------------------------------------------

    N_CLIMA_FEATURES = VETOR_CLIMA.shape[0]

    print(f"Vetor de Clima Normalizado (usado para cada amostra): {VETOR_CLIMA}")
    
    print("\n--- 2. Preparação do Dataset de Imagens ---")
    dataset = load_dataset("samokosik/clothes_simplified")
    df_train = dataset['train']

    processados = [extract_features(df_train[i]) for i in range(len(df_train))]
    processados = [p for p in processados if p[0] is not None]

    X_list_img, Y_list, JSON_list = zip(*processados)


    X_img = torch.tensor(np.array(X_list_img), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.long)
    
    # Cria o vetor de clima repetido
    VETOR_CLIMA_REPETIDO = np.tile(VETOR_CLIMA, (X_img.shape[0], 1))
    X_clima = torch.tensor(VETOR_CLIMA_REPETIDO, dtype=torch.float32)

    # Concatena as features da imagem e do clima
    X_COMBINADO = torch.cat((X_img, X_clima), dim=1)
    
    N_ENTRADAS = X_COMBINADO.shape[1]
    N_SAIDAS = len(np.unique(Y.numpy()))

    print("\n--- 3. Resumo da Combinação ---")
    print(f"Entradas da Imagem (Pixels): {IMG_FLAT_DIM}")
    print(f"Entradas do Clima: {N_CLIMA_FEATURES}")
    print(f"Entradas Combinadas (N_ENTRADAS): {N_ENTRADAS}")
    print(f"Saídas (N_SAIDAS/Classes): {N_SAIDAS}")
    print(f"Tensor de Entrada Combinado X_COMBINADO.shape: {X_COMBINADO.shape}")
    
    return X_COMBINADO, Y, N_ENTRADAS, N_SAIDAS

if __name__ == '__main__':
    # Exemplo de uso
    X_COMBINADO, Y, N_ENTRADAS, N_SAIDAS = preparar_dados_combinados()