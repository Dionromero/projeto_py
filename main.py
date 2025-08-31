#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

x, y = make_blobs(random_state=1)

plt.scatter(x[:, 0], x[:, 1])

#criar dataframe para visualizar como tabela 
df = pd.DataFrame(x, columns=['feature_1', 'feature_2'])
df ['cluster'] = y #adiciona a coluna cluster 

#mostrar as primeiras linhas da tabela no terminal 
print("Primeiras 10 linhas do terminal:")
print(df.head(10))