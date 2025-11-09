# clima_api.py

import pandas as pd
import requests
import numpy as np

# Configurações do pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# --- Funções de Clima ---

def obter_dados_clima(local="Curitiba"):
    """Busca dados climáticos da API e retorna o JSON."""
    API_KEY = "8edf5a3557214b83a8d24412250109" 
    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": API_KEY,
        "q": local,
        "lang": "pt",
        "aqi": "no"
    }
    
    try:
        # Tenta obter dados reais
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"⚠️ Alerta: Erro ao obter dados climáticos: {e}. Usando dados simulados.")
        # Retorna dados simulados para garantir que o resto do código funcione
        return {
            "current": {"temp_c": 22.5, "humidity": 75, "is_day": 1},
            "forecast": {"forecastday": [{"hour": [{"time": "2025-11-09 12:00", "temp_c": 22.5, "humidity": 75, "chance_of_rain": 20}]}]}
        }

def criar_dataframes(dados):
    """Cria DataFrames principais e de previsão a partir do JSON."""
    if dados is None:
        return None, None
    df = pd.json_normalize(dados)
    
    if "forecast" in dados and "forecastday" in dados["forecast"]:
        forecast_list = dados["forecast"]["forecastday"]
        df_forecast = pd.json_normalize(forecast_list)
    else:
        df_forecast = pd.DataFrame()
        
    return df, df_forecast

def processar_dados_horarios(df_forecast):
    """Extrai, normaliza e retorna o vetor climático atual (Temp, Umid, Chuva)."""
    # Simplificado: Extraímos apenas os dados do 'current' para usar um único vetor de clima
    
    # Retorna vetor padrão se falhar
    if df_forecast is None or 'hour' not in df_forecast.columns or len(df_forecast) == 0:
        return np.array([20.0/40.0, 70.0/100.0, 10.0/100.0], dtype=np.float32) 
    
    # Seu código começa aqui:
    hour_list = df_forecast['hour'].iloc[0]
    df_hour = pd.json_normalize(hour_list)
    df_hour["time"] = pd.to_datetime(df_hour["time"])
    df_hour = df_hour.set_index("time")
    df_numeric = df_hour.select_dtypes(include="number")
    # Usamos o primeiro registro (previsão atual/próxima) como o vetor de clima
    df_current_hour = df_numeric.iloc[0][['temp_c', 'humidity', 'chance_of_rain']] 
    
    # Normalização dos dados de clima:
    # 1. Temperatura: 0°C a 40°C
    # 2. Umidade: 0 a 100
    # 3. Chuva: 0 a 100 (chance_of_rain)
    df_current_hour['temp_c'] = df_current_hour['temp_c'] / 40.0
    df_current_hour['humidity'] = df_current_hour['humidity'] / 100.0
    df_current_hour['chance_of_rain'] = df_current_hour['chance_of_rain'] / 100.0
    
    return df_current_hour.to_numpy().astype(np.float32)