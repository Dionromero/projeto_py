import pandas as pd
import requests

# Configurações do pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Função para obter dados climáticos
def obter_dados_clima(local="Curitiba"):
    API_KEY = "8edf5a3557214b83a8d24412250109"
    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": API_KEY,
        "q": local,
        "lang": "pt",
        "aqi": "no"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        dados = response.json()
        return dados
    except Exception as e:
        print(f"Erro ao obter dados climáticos: {e}")
        return None

# Função para criar dataframes
def criar_dataframes(dados):
    if dados is None:
        return None, None
    
    # Dataframe principal
    df = pd.json_normalize(dados)
    
    # Dataframe de previsão
    forecast_list = dados["forecast"]["forecastday"]
    df_forecast = pd.json_normalize(forecast_list)
    
    return df, df_forecast

# Função para processar dados horários
def processar_dados_horarios(df_forecast):
    if df_forecast is None or len(df_forecast) == 0:
        return None
    
    # Extrair a lista de horas do primeiro dia de previsão
    hour_list = df_forecast['hour'][0]
    
    # Normalizar a lista para criar o dataframe de horas
    df_hour = pd.json_normalize(hour_list)
    
    # Converter para datetime
    df_hour["time"] = pd.to_datetime(df_hour["time"])
    
    # Definir como índice
    df_hour = df_hour.set_index("time")
    
    # Selecionar apenas colunas numéricas
    df_numeric = df_hour.select_dtypes(include="number")
    
    # Reamostrar de 5h em 5h e calcular média
    df_5h = df_numeric.resample("5h").mean().reset_index()
    
    return df_5h

# Executar o código e criar as variáveis globais
dados = obter_dados_clima()
df, df_forecast = criar_dataframes(dados)
df_5h = processar_dados_horarios(df_forecast)