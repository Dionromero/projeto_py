import pandas as pd
dados = {
    'cliente_id': [1, 2, 3, 4, 5],
    'idade': [25, 40, 30, 22, 35],
    'estilo': ['casual', 'social', 'esportivo', 'casual', 'social'],
    'temperatura': [30, 12, 22, 28, 15],
    'chuva': [10, 80, 20, 5, 50],
    'vento': [5, 20, 10, 3, 15],
    'roupa_recomendada': ['camiseta', 'casaco', 'camisa', 'camiseta', 'jaqueta']
}
df_clientes = pd.DataFrame(dados)
print(df_clientes)