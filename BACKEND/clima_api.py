import pandas as pd
import requests
import numpy as np

# Chave de API
API_KEY = "8edf5a3557214b83a8d24412250109"

def obter_dados_clima(local="Curitiba"):
    url = "http://api.weatherapi.com/v1/forecast.json"
    try:
        r = requests.get(url, params={"key": API_KEY, "q": local, "lang": "pt", "days": 1}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Erro API: {e}")
        # Fallback (Dados falsos caso a API falhe)
        return {
            "current": {"temp_c": 20.0, "humidity": 50, "condition": {"text": "Nublado"}},
            "forecast": {"forecastday": [{
                "hour": [{
                        "time": f"2025-01-01 {h:02d}:00",
                        "temp_c": 20,
                        "humidity": 60,
                        "chance_of_rain": 50, # Teste: forçando chuva no fallback
                        "wind_kph": 15,
                        "condition": {"text": "Chuva"}
                    } for h in range(24)]
            }]}
        }

def criar_dataframes(dados):
    if not dados: return None, None
    df_forecast = (pd.json_normalize(dados["forecast"]["forecastday"]) if "forecast" in dados else pd.DataFrame())
    return pd.json_normalize(dados), df_forecast

def processar_dados_horarios(df_forecast):
    try:
        hour_list = df_forecast["hour"].iloc[0]
        df_hour = pd.json_normalize(hour_list)
        row = df_hour.iloc[0]
        return np.array([row["temp_c"]/40, row["humidity"]/100, row["chance_of_rain"]/100], dtype=np.float32)
    except:
        return np.array([0.5, 0.5, 0.1], dtype=np.float32)

# --- CORREÇÃO PRINCIPAL AQUI ---
def processar_dados_para_tabela(dados):
    if not dados or "forecast" not in dados: return []
    lista_final = []
    try:
        horas = dados["forecast"]["forecastday"][0]["hour"]
        for h in horas:
            # Força conversão para INTEIRO (int) para garantir que não venha como string
            chuva = int(h.get("chance_of_rain", 0))
            vento = float(h.get("wind_kph", 0))
            
            item = {
                "timestamp": h["time"],
                "temperatura": h["temp_c"],
                "umidade": h["humidity"],
                "chance_of_rain": chuva, # Agora é garantido ser número
                "vento_kph": vento,
                "condicao": h["condition"]["text"]
            }
            lista_final.append(item)
    except Exception as e:
        print(f"Erro processamento: {e}")
        return []
    return lista_final