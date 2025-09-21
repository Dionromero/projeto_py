# Arquivo: modelagem.ipynb

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def treinar_modelo_estilos():
    """
    Treina e retorna um modelo de machine learning para prever estilos de roupa.
    """
    # Dados de treinamento expandidos com diversas variações de clima
    data = {
        'estilos': [
            'casual', 'casual', 'casual', 'casual', 'casual',
            'esportivo', 'esportivo', 'esportivo', 'esportivo', 'esportivo',
            'social', 'social', 'social',
            'streetwear', 'streetwear', 'streetwear',
            'frio', 'frio', 'frio',
            'calor', 'calor', 'calor'
        ],
        'data e hora': [
            '2025-09-18 00:00', '2025-10-05 15:00', '2025-11-20 09:00', '2026-03-10 12:00', '2025-05-20 18:00',
            '2025-09-18 01:00', '2025-10-10 10:00', '2025-11-25 18:00', '2026-04-01 08:00', '2025-06-15 17:00',
            '2025-09-18 02:00', '2025-10-12 20:00', '2026-01-20 21:00',
            '2025-09-18 03:00', '2025-11-01 14:00', '2026-02-28 16:00',
            '2025-07-10 10:00', '2025-08-22 08:00', '2025-12-05 22:00',
            '2026-01-15 13:00', '2026-02-05 16:00', '2025-04-30 11:00'
        ],
        'temperatura': [
            13.1, 20.5, 22.0, 25.0, 18.0,
            12.8, 25.0, 28.0, 22.0, 15.0,
            13.0, 18.0, 30.0,
            12.8, 15.0, 28.0,
            5.0, 8.0, 10.0,
            30.0, 32.5, 29.0
        ],
        'chance de chuva': [
            86, 20, 10, 5, 45,
            74, 5, 0, 15, 60,
            71, 15, 0,
            0, 30, 0,
            60, 50, 75,
            0, 0, 10
        ],
        'umidade': [
            84, 60, 55, 50, 70,
            83, 50, 40, 60, 80,
            81, 70, 35,
            82, 75, 40,
            90, 88, 92,
            30, 25, 32
        ],
        'uv': [
            0.0, 5.0, 6.0, 8.0, 3.0,
            0.0, 8.0, 9.0, 7.0, 2.0,
            0.0, 2.0, 9.0,
            0.0, 1.0, 9.0,
            0.0, 0.0, 0.0,
            10.0, 10.0, 8.0
        ],
        'condição do dia': [
            'Possibilidade de chuva irregular', 'Ensolarado', 'Parcialmente nublado', 'Céu limpo', 'Nublado',
            'Possibilidade de chuva irregular', 'Céu limpo', 'Ensolarado', 'Parcialmente nublado', 'Chuvoso',
            'Possibilidade de chuva irregular', 'Nublado', 'Céu limpo',
            'Nublado', 'Possibilidade de chuva irregular', 'Ensolarado',
            'Chuvoso', 'Neblina', 'Chuvoso',
            'Céu limpo', 'Ensolarado', 'Céu limpo'
        ]
    }

    df_tabela1 = pd.DataFrame(data)
    df_tabela1['data e hora'] = pd.to_datetime(df_tabela1['data e hora'])
    df_tabela1['data'] = df_tabela1['data e hora'].dt.date
    df_tabela1['hora'] = df_tabela1['data e hora'].dt.hour
    df_tabela1['estacao_do_ano'] = df_tabela1['data'].apply(
        lambda x: 'verao' if (x.month > 11 or x.month < 3)
        else 'outono' if (x.month > 2 and x.month < 6)
        else 'inverno' if (x.month > 5 and x.month < 9)
        else 'primavera'
    )

    df_dummies = pd.get_dummies(df_tabela1[['condição do dia', 'estacao_do_ano', 'hora']])
    X = df_dummies
    y = df_tabela1['estilos']

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    # Retorna o modelo treinado e as colunas (features) usadas para o treinamento
    return clf, X.columns

# Se este arquivo for executado diretamente, treine o modelo.
if __name__ == '__main__':
    modelo_treinado, colunas_modelo = treinar_modelo_estilos()
    print("Modelo treinado com sucesso!")
    print(f"Colunas usadas para o treinamento: {colunas_modelo.tolist()}")
