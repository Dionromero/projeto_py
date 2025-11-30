import sys
import os
from flask import Blueprint, jsonify

# --- CORREÇÃO: Ajuste de caminho ANTES dos imports ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -----------------------------------------------------

from clima_api import obter_dados_clima, processar_dados_para_tabela

weather_bp = Blueprint('weather', __name__)

@weather_bp.route('/', methods=['GET'])
def get_weather():
    dados = obter_dados_clima("Curitiba")
    tabela = processar_dados_para_tabela(dados)
    return jsonify(tabela)