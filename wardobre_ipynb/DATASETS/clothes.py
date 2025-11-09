
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

