# motor_recomendacao.py
import numpy as np
# Mapeia os IDs/Nomes do seu LABEL_MAP para regras de negócio
# LABEL_MAP original: 0: "Camiseta", 1: "Calça", 2: "Vestido", 3: "Jaqueta", 
# 4: "Saia", 5: "Short", 6: "Suéter", 7: "Blusa", 8: "Meia", 
# 9: "Sapato", 10: "Chapéu", 11: "Acessório"

CATALOGO_REGRAS = {
    "Camiseta": {"parte": "tronco", "genero": ["unisex"], "clima": ["quente", "neutro"]},
    "Calça":    {"parte": "pernas", "genero": ["unisex"], "clima": ["neutro", "frio"]},
    "Vestido":  {"parte": "corpo_inteiro", "genero": ["female"], "clima": ["quente", "neutro"]},
    "Jaqueta":  {"parte": "tronco_externo", "genero": ["unisex"], "clima": ["frio", "muito_frio"]},
    "Saia":     {"parte": "pernas", "genero": ["female"], "clima": ["quente"]},
    "Short":    {"parte": "pernas", "genero": ["unisex"], "clima": ["quente"]},
    "Suéter":   {"parte": "tronco", "genero": ["unisex"], "clima": ["frio"]},
    "Blusa":    {"parte": "tronco", "genero": ["female"], "clima": ["neutro", "quente"]},
    "Meia":     {"parte": "pes_interno", "genero": ["unisex"], "clima": ["neutro", "frio"]},
    "Sapato":   {"parte": "pes", "genero": ["unisex"], "clima": ["neutro"]},
    "Chapéu":   {"parte": "cabeca", "genero": ["unisex"], "clima": ["quente"]},
    "Acessório":{"parte": "acessorios", "genero": ["unisex"], "clima": ["neutro"]}
}

def definir_tipo_clima(temp_c):
    """Traduz temperatura numérica para categoria."""
    if temp_c < 16: return "muito_frio"
    if 16 <= temp_c < 20: return "frio"
    if 20 <= temp_c < 26: return "neutro"
    return "quente"

def filtrar(todas_classes, temperatura, genero_usuario):
    """
    Recebe todas as classes possíveis e retorna apenas as que
    fazem sentido para o clima e gênero atuais, organizadas por parte do corpo.
    """
    clima_atual = definir_tipo_clima(temperatura)
    
    recomendacao_estruturada = {
        "cabeca": [],
        "tronco": [],
        "tronco_externo": [], # Casacos/Jaquetas
        "pernas": [],
        "pes": [],
        "corpo_inteiro": []
    }

    for nome_roupa, regras in CATALOGO_REGRAS.items():
        # 1. Verifica Gênero
        if "unisex" in regras["genero"] or genero_usuario in regras["genero"]:
            
            # 2. Verifica Clima (Lógica flexível: aceita clima exato ou adjacentes)
            # Ex: Se está "frio", aceita roupas de "frio" e "muito_frio"
            aceita_clima = False
            if clima_atual in regras["clima"]:
                aceita_clima = True
            elif clima_atual == "frio" and "muito_frio" in regras["clima"]:
                aceita_clima = True # Jaqueta serve no frio
            elif clima_atual == "neutro" and "quente" in regras["clima"]:
                aceita_clima = True # Camiseta serve no neutro
            
            if aceita_clima:
                parte = regras["parte"]
                if parte in recomendacao_estruturada:
                    recomendacao_estruturada[parte].append(nome_roupa)

    return recomendacao_estruturada, clima_atual