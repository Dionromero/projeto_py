import sys
import os
from flask import Blueprint, jsonify, request

# Ajuste de caminho
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clima_api import obter_dados_clima
from motor_recomendacao import filtrar, CATALOGO_REGRAS

recommendations_bp = Blueprint('recommendations', __name__)

LABEL_MAP = {
    0: "Camiseta", 1: "Calça", 2: "Vestido", 3: "Jaqueta", 
    4: "Saia", 5: "Short", 6: "Suéter", 7: "Blusa", 
    8: "Meia", 9: "Sapato", 10: "Chapéu", 11: "Acessório"
}

@recommendations_bp.route('/', methods=['GET'])
def recomendar():
    # 1. Captura parâmetros (genero agora vem do front)
    genero = request.args.get('genero', 'unisex')
    local = request.args.get('local', 'Curitiba')

    # 2. Clima
    try:
        dados_clima = obter_dados_clima(local)
        temp_atual = dados_clima['current']['temp_c']
    except:
        temp_atual = 25.0 

    # 3. Filtra as roupas
    todas_roupas = list(LABEL_MAP.values())
    sugestoes_por_parte, categoria_clima = filtrar(todas_roupas, temp_atual, genero)

    # 4. Monta o Look Completo (Pega o 1º item disponível de cada lista)
    look = {
        "cabeca": sugestoes_por_parte['cabeca'][0] if sugestoes_por_parte['cabeca'] else "Nada",
        "tronco": sugestoes_por_parte['tronco'][0] if sugestoes_por_parte['tronco'] else "Nada",
        "casaco": sugestoes_por_parte['tronco_externo'][0] if sugestoes_por_parte['tronco_externo'] else None,
        "pernas": sugestoes_por_parte['pernas'][0] if sugestoes_por_parte['pernas'] else "Nada",
        "pes": sugestoes_por_parte['pes'][0] if sugestoes_por_parte['pes'] else "Nada"
    }

    # Se tiver casaco, substituimos ou adicionamos ao tronco para visualização
    peca_superior = look['casaco'] if look['casaco'] else look['tronco']
    
    # Tratamento especial para vestido (cobre tronco e pernas)
    if sugestoes_por_parte['corpo_inteiro']:
        look['tronco'] = sugestoes_por_parte['corpo_inteiro'][0]
        look['pernas'] = "(Peça Única)"

    resposta = {
        "status": "success",
        "local": local,
        "temperatura": temp_atual,
        "categoria_clima": categoria_clima,
        "look": look, # Objeto novo com todas as partes
        "imagem_path": f"static/images/{peca_superior.lower()}.jpg" # Imagem destaque
    }

    return jsonify(resposta)