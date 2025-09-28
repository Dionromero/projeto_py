# %%
from datasets import load_dataset
from PIL import Image
import numpy as np

# %%
# Carrega o dataset 'clothes_simplified'
dataset = load_dataset("samokosik/clothes_simplified")

# %%
# Exibe informações do dataset
print(dataset)

# %%
# Acesso aos splits
df_train = dataset['train']
df_test = dataset['test']

# %%
# Exemplo de acesso a um item
print(df_train[0])

# %%
# Função para extrair features simples de imagem (opcional)

def extract_features(img_path, size=(32,32)):
    img = Image.open(img_path).resize(size)
    return np.array(img).flatten()
