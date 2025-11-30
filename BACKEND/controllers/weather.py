import sys
import os
from flask import Blueprint, jsonify

# Ajuste de caminho
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from clima_api import obter_dados_clima

weather_bp = Blueprint('weather', __name__)

@weather_bp.route('/', methods=['GET'])
def get_weather():
    # Agora retornamos os DADOS COMPLETOS (Current + Forecast)
    # Não processamos para tabela aqui, deixamos o Javascript fazer isso
    dados = obter_dados_clima("Curitiba")
    
    # Se der erro no clima_api e voltar None, mandamos um erro JSON
    if not dados:
        return jsonify({"error": "Clima indisponível"}), 503
        
    return jsonify(dados)