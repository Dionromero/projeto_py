from flask import Blueprint, jsonify
from clima_api import obter_dados_clima, processar_dados_para_tabela
import sys
import os
# Adiciona o diretório pai ao caminho do sistema para encontrar 'clima_api' e 'motor_recomendacao'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

weather_bp = Blueprint('weather', __name__)

@weather_bp.route('/', methods=['GET'])
def get_weather():
    # Usa a sua função original do clima_api.py
    dados = obter_dados_clima("Curitiba")
    tabela = processar_dados_para_tabela(dados)
    return jsonify(tabela)