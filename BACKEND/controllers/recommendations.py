import torch
from flask import Blueprint, jsonify, request
from clima_api import obter_dados_clima
from motor_recomendacao import filtrar, CATALOGO_REGRAS
import sys
import os
# Adiciona o diretório pai ao caminho do sistema para encontrar 'clima_api' e 'motor_recomendacao'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

recommendations_bp = Blueprint('recommendations', __name__)

# --- CONFIGURAÇÃO DO LABEL MAP (Sincronizado com motor_recomendacao) ---
LABEL_MAP = {
    0: "Camiseta", 1: "Calça", 2: "Vestido", 3: "Jaqueta", 
    4: "Saia", 5: "Short", 6: "Suéter", 7: "Blusa", 
    8: "Meia", 9: "Sapato", 10: "Chapéu", 11: "Acessório"
}

@recommendations_bp.route('/', methods=['GET'])
def recomendar():
    # 1. Captura parâmetros da URL (Ex: /recomendar?genero=female&local=Curitiba)
    genero = request.args.get('genero', 'unisex') # Padrão unisex
    local = request.args.get('local', 'Curitiba')

    # 2. Obtém dados reais do clima
    try:
        dados_clima = obter_dados_clima(local)
        temp_atual = dados_clima['current']['temp_c']
        condicao = dados_clima['current']['condition']['text']
    except:
        temp_atual = 25.0 # Fallback
        condicao = "Ensolarado"

    # 3. Gera a recomendação baseada em regras (Lógica Nova)
    # Pegamos todas as roupas possíveis (valores do LABEL_MAP) e filtramos
    todas_roupas = list(LABEL_MAP.values())
    sugestoes_por_parte, categoria_clima = filtrar(todas_roupas, temp_atual, genero)

    # 4. (Opcional) Integração com o Modelo AI
    # Se você tivesse um banco de dados de imagens do usuário, aqui você:
    # a) Pegaria as imagens do banco.
    # b) Passaria no ModeloVisaoClimaLite para confirmar a categoria.
    # c) Filtraria usando as sugestoes_por_parte acima.
    
    # Como estamos sugerindo o que vestir (sem inventário do usuário ainda):
    resposta = {
        "status": "success",
        "contexto": {
            "local": local,
            "temperatura": temp_atual,
            "sensacao_termica": categoria_clima, # ex: "frio", "quente"
            "condicao": condicao,
            "genero_selecionado": genero
        },
        "recomendacoes": sugestoes_por_parte
    }

    return jsonify(resposta)