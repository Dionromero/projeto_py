import sys
import os
from flask import Blueprint, jsonify, request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clima_api import obter_dados_clima
from motor_recomendacao import filtrar

recommendations_bp = Blueprint('recommendations', __name__)

@recommendations_bp.route('/', methods=['GET'])
def recomendar():
    genero = request.args.get('genero', 'unisex')
    local = request.args.get('local', 'Curitiba')

    # Clima
    try:
        dados_clima = obter_dados_clima(local)
        temp_atual = dados_clima['current']['temp_c']
    except:
        temp_atual = 25.0 

    # IMPORTANTE: Passamos None aqui, pois o motor_recomendacao agora
    # itera sobre o seu próprio CATALOGO_REGRAS interno, não dependendo
    # apenas da lista limitada do LabelMap antigo.
    sugestoes_por_parte, categoria_clima = filtrar(None, temp_atual, genero)

    # Monta o look pegando o primeiro item disponível de cada lista
    # Adicionamos fallbacks ("-") caso ainda assim não encontre nada
    look = {
        "cabeca": sugestoes_por_parte['cabeca'][0] if sugestoes_por_parte['cabeca'] else "-",
        "tronco": sugestoes_por_parte['tronco'][0] if sugestoes_por_parte['tronco'] else "-",
        "casaco": sugestoes_por_parte['tronco_externo'][0] if sugestoes_por_parte['tronco_externo'] else None,
        "pernas": sugestoes_por_parte['pernas'][0] if sugestoes_por_parte['pernas'] else "-",
        "pes": sugestoes_por_parte['pes'][0] if sugestoes_por_parte['pes'] else "-"
    }

    if sugestoes_por_parte['corpo_inteiro']:
        look['tronco'] = sugestoes_por_parte['corpo_inteiro'][0]
        look['pernas'] = "(Peça Única)"

    peca_destaque = look['casaco'] if look['casaco'] else look['tronco']
    
    return jsonify({
        "status": "success",
        "local": local,
        "temperatura": temp_atual,
        "categoria_clima": categoria_clima,
        "look": look,
        "imagem_path": f"static/images/{str(peca_destaque).lower()}.jpg"
    })