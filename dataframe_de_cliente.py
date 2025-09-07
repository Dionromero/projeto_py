import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

n_clientes = 20

def recomendar_roupas (temp, estilo):
    if temp <= 10:
        return np.random.choice(['Casaco', 'Jaqueta', 'Suéter'])
    elif 10 < temp <= 20:
        if estilo == 'social':
            return 'camisa'
        else:
            return np.random.choice(['Camiseta', 'Camisa de Manga Longa'])
    else: # quando a temperatura estiver quente
        if temp >=20:
            if estilo == 'social':
                return 'Camisa de Manga Curta'
            else:
                return np.random.choice(['Regata', 'Camiseta'])

def recomendar_acessorios (temp, estilo, uv):
    if temp <= 10:
        return np.random.choice(['Cachecol', 'Luvas', 'Gorro'])
    elif 10 < temp <= 20:
        if estilo == 'social':
            return 'Relógio'
    elif uv >= 6:
        return np.random.choice(['Boné', 'Óculos de Sol'])
    else: # quando a temperatura estiver quente
        if temp >=20:
            if estilo == 'social':
                return 'Chapéu'
            else:
                return np.random.choice(['Pulseira', 'Colar'])            
            
idade = np.random.randint(18, 60, n_clientes)
estilo = np.random.choice(['casual', 'social'], n_clientes)
temperaturas = np.random.randint(5, 35, n_clientes)  # Adiciona definição de temperaturas
uv = np.random.randint(1, 11, n_clientes)  # Adiciona definição de uv

roupas = [recomendar_roupas(temp, est)
          for temp, est in zip(temperaturas, estilo)]
acessorios = [recomendar_acessorios(temp, est, u)
                for temp, est, u in zip(temperaturas, estilo, uv)]

df_clientes = pd.DataFrame({
    'idade': idade,
    'estilo': estilo,
    'roupa_recomendada': roupas,
    'acessorio_recomendado': acessorios
})
def aplicar_recomendações(df_clientes, df_limpo):
    df_recomendacoes = df_clientes.copy()
    df_recomendacoes = df_recomendacoes.merge(df_limpo, how='left', left_on='roupa_recomendada', right_on='roupa')
    df_recomendacoes = df_recomendacoes.merge(df_limpo, how='left', left_on='acessorio_recomendado', right_on='roupa', suffixes=('_roupa', '_acessorio'))
    
    df_recomendacoes = df_recomendacoes[['idade', 'estilo', 'roupa_recomendada', 'acessorio_recomendado',]]
    return df_recomendacoes
print(df_clientes)
print(df_clientes.info())
