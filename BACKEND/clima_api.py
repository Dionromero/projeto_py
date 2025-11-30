import pandas as pd
import requests
import numpy as np

# Sua chave (idealmente use variáveis de ambiente em produção)
API_KEY = "8edf5a3557214b83a8d24412250109"

def obter_dados_clima(local="Curitiba"):
    """Busca clima ou retorna fallback sem travar o sistema."""
    url = "http://api.weatherapi.com/v1/forecast.json"

    try:
        r = requests.get(url, params={"key": API_KEY, "q": local, "lang": "pt", "days": 1}, timeout=5)
        r.raise_for_status()
        return r.json()

    except Exception as e:
        print(f"Erro na API de Clima: {e}")
        # Fallback seguro para não quebrar o site se a API falhar
        return {
            "current": {"temp_c": 20.0, "humidity": 50, "condition": {"text": "Nublado"}},
            "forecast": {"forecastday": [{
                "hour": [
                    {
                        "time": f"2025-01-01 {h:02d}:00",
                        "temp_c": 20 + (h % 5),
                        "humidity": 60,
                        "chance_of_rain": 10,
                        "wind_kph": 15,
                        "condition": {"text": "Nublado"}
                    } for h in range(24)
                ]
            }]}
        }

def criar_dataframes(dados):
    """Normaliza JSON em DataFrames (Usado pelo treinamento)."""
    if not dados:
        return None, None

    df_forecast = (
        pd.json_normalize(dados["forecast"]["forecastday"])
        if "forecast" in dados
        else pd.DataFrame()
    )

    return pd.json_normalize(dados), df_forecast


def processar_dados_horarios(df_forecast):
    """Extrai vetor [temp, humidity, rain] normalizado para a Rede Neural."""
    try:
        hour_list = df_forecast["hour"].iloc[0]
        df_hour = pd.json_normalize(hour_list)

        row = df_hour.iloc[0]
        temp = row["temp_c"] / 40
        hum = row["humidity"] / 100
        rain = row["chance_of_rain"] / 100

        return np.array([temp, hum, rain], dtype=np.float32)

    except:
        return np.array([0.5, 0.5, 0.1], dtype=np.float32)

# --- FUNÇÃO PRINCIPAL USADA PELA TABELA DO SITE ---
def processar_dados_para_tabela(dados):
    """Extrai e formata dados detalhados para o Frontend (Tailwind)."""
    if not dados or "forecast" not in dados:
        return []

    lista_final = []

    try:
        # Pega a lista de horas do primeiro dia de previsão
        horas = dados["forecast"]["forecastday"][0]["hour"]

        for h in horas:
            item = {
                "timestamp": h["time"],          # Data/Hora original
                "temperatura": h["temp_c"],      # Temperatura
                "umidade": h["humidity"],        # Umidade
                "chance_of_rain": h.get("chance_of_rain", 0), # Chuva (%)
                "vento_kph": h.get("wind_kph", 0),            # Vento (km/h)
                "condicao": h["condition"]["text"]            # Texto (ex: Sol)
            }
            lista_final.append(item)
            
    except Exception as e:
        print(f"Erro ao processar tabela: {e}")
        return []

    return lista_final