import pandas as pd
import numpy as np
from datasets import load_dataset     
  
# Carrega o dataset 'clothes_simplified'
dataset = load_dataset("samokosik/clothes_simplified")
# Exibe informações do dataset
print(dataset)
# Acesso aos splits
df_train = dataset['train']
df_test = dataset['test']
# Exemplo de acesso a um item
print(df_train[0])

df_train = dataset["train"].to_pandas()
df_test = dataset["test"].to_pandas()

print(df_train.head())

categorias_numericas = df_train["label"].unique()
print("Categorias numéricas:", categorias_numericas)

tabela_categorias = (
    df_train[["label"]]
    .drop_duplicates()
    .sort_values("label")
)

pd.set_option('display.max_rows', None)
print(tabela_categorias)

# normalizando os rotulos para começar de 0
classes = np.unique(dataset['train']['label'])
classes

label_map = {old: new for new, old in enumerate(classes)}
label_map

def normalize_labels(example):
    example['label'] = label_map[example['label']]
    return example

dataset = dataset.map(normalize_labels)

print(np.unique(dataset['train']['label']))
print(np.unique(dataset['test']['label']))
