import pandas as pd
import requests
import numpy as np


def obter_dados_clima(local="Curitiba"):
    """Busca clima ou retorna fallback sem travar o sistema."""
    API_KEY = "8edf5a3557214b83a8d24412250109"
    url = "http://api.weatherapi.com/v1/forecast.json"

    try:
        r = requests.get(url, params={"key": API_KEY, "q": local, "lang": "pt"}, timeout=3)
        r.raise_for_status()
        return r.json()

    except Exception:
        # fallback seguro
        return {
            "current": {"temp_c": 22.5, "humidity": 70},
            "forecast": {"forecastday": [{
                "hour": [{
                    "time": "2025-01-01 12:00",
                    "temp_c": 22.5,
                    "humidity": 70,
                    "chance_of_rain": 10
                }]
            }]}
        }


def criar_dataframes(dados):
    """Normaliza JSON em DataFrames."""
    if not dados:
        return None, None

    df_forecast = (
        pd.json_normalize(dados["forecast"]["forecastday"])
        if "forecast" in dados
        else pd.DataFrame()
    )

    return pd.json_normalize(dados), df_forecast


def processar_dados_horarios(df_forecast):
    """Extrai vetor [temp, humidity, rain] normalizado."""

    try:
        hour_list = df_forecast["hour"].iloc[0]
        df_hour = pd.json_normalize(hour_list)

        row = df_hour.iloc[0]
        temp = row["temp_c"] / 40
        hum = row["humidity"] / 100
        rain = row["chance_of_rain"] / 100

        return np.array([temp, hum, rain], dtype=np.float32)

    except:  # noqa: E722
        return np.array([0.5, 0.5, 0.1], dtype=np.float32)

def processar_dados_para_tabela(dados):
    """Extrai e formata dados de previsão horária para o frontend."""
    if not dados or "forecast" not in dados:
        return []

    try:
        hour_list = dados["forecast"]["forecastday"][0]["hour"]
        # Filtra e formata os dados necessários
        tabela_dados = []
        for hora in hour_list:
            tabela_dados.append({
                "timestamp": hora["time"],
                "temperatura": hora["temp_c"],
                "umidade": hora["humidity"],
                "condicao": hora["condition"]["text"],
            })
        return tabela_dados
    except Exception:
        return []