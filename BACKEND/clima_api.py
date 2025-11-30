import requests
import pandas as pd
import numpy as np

API_KEY = "8edf5a3557214b83a8d24412250109"

def obter_dados_clima(local="Curitiba"):
    url = "http://api.weatherapi.com/v1/forecast.json"
    try:
        r = requests.get(url, params={"key": API_KEY, "q": local, "lang": "pt", "days": 1}, timeout=3)
        r.raise_for_status()
        return r.json()
    except:
        return None

def processar_dados_para_tabela(dados):
    if not dados or "forecast" not in dados: return []
    lista_final = []
    try:
        horas = dados["forecast"]["forecastday"][0]["hour"]
        for h in horas:
            item = {
                "timestamp": h["time"],
                "temperatura": h["temp_c"],
                "umidade": h["humidity"],
                "chance_of_rain": int(h.get("chance_of_rain", 0)),
                "vento_kph": float(h.get("wind_kph", 0)),
                "condicao": h["condition"]["text"]
            }
            lista_final.append(item)
    except:
        return []
    return lista_final