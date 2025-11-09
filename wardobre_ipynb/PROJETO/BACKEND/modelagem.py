from datasets import load_dataset
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import clima as clima 

# 1. Carregar e Pré-processar as Imagens e Rótulos
dataset = load_dataset("samokosik/clothes_simplified")
df_train = dataset['train']

def extract_features(item, size=(32, 32)):
    # Acessa o objeto de imagem PIL diretamente do item do dataset
    img = item['image'].resize(size)
    # Normaliza e achata: [32, 32, 3] -> [3072]
    # Normalizamos por 255 para ter valores entre 0 e 1, o que é comum em NN
    features = np.array(img, dtype=np.float32).flatten() / 255.0
    return features, item['label']

# Extrai todas as features e rótulos
# Observação: A coluna de rótulo pode se chamar 'label' ou 'class_id'
X_list, Y_list = zip(*[extract_features(df_train[i]) for i in range(len(df_train))])

# Converte para Tensores
X = torch.tensor(np.array(X_list), dtype=torch.float32)
Y = torch.tensor(np.array(Y_list), dtype=torch.long)

# Define o número de entradas e saídas
N_ENTRADAS = X.shape[1] # 3072 (32*32*3)
N_SAIDAS = len(np.unique(Y.numpy())) # Número de classes de roupa







print("Treinamento concluído ✅")

print(clima.df_5h.head())  

