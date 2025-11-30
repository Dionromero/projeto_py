# motor_recomendacao.py

# Expandimos o catálogo para garantir cobertura total
CATALOGO_REGRAS = {
    # TRONCO
    "Camiseta": {"parte": "tronco", "genero": ["unisex"], "clima": ["quente", "neutro"]},
    "Suéter":   {"parte": "tronco", "genero": ["unisex"], "clima": ["frio", "muito_frio"]},
    "Blusa":    {"parte": "tronco", "genero": ["female"], "clima": ["neutro", "quente"]},
    "Vestido":  {"parte": "corpo_inteiro", "genero": ["female"], "clima": ["quente", "neutro"]},
    
    # CASACO
    "Jaqueta":  {"parte": "tronco_externo", "genero": ["unisex"], "clima": ["frio", "muito_frio"]},

    # PERNAS
    "Calça":    {"parte": "pernas", "genero": ["unisex"], "clima": ["neutro", "frio", "muito_frio"]},
    "Short":    {"parte": "pernas", "genero": ["unisex"], "clima": ["quente"]},
    "Saia":     {"parte": "pernas", "genero": ["female"], "clima": ["quente", "neutro"]},

    # --- NOVOS ITENS PARA CABEÇA E PÉS (Garantia de Preenchimento) ---
    
    # CABEÇA
    "Boné":     {"parte": "cabeca", "genero": ["unisex"], "clima": ["quente", "neutro"]},
    "Chapéu":   {"parte": "cabeca", "genero": ["unisex"], "clima": ["quente"]},
    "Gorro":    {"parte": "cabeca", "genero": ["unisex"], "clima": ["frio", "muito_frio"]},
    
    # PÉS
    "Tênis":    {"parte": "pes", "genero": ["unisex"], "clima": ["quente", "neutro", "frio"]},
    "Bota":     {"parte": "pes", "genero": ["unisex"], "clima": ["frio", "muito_frio"]},
    "Sandália": {"parte": "pes", "genero": ["female", "unisex"], "clima": ["quente"]},
    "Sapato":   {"parte": "pes", "genero": ["unisex"], "clima": ["neutro"]}
}

def definir_tipo_clima(temp):
    if temp >= 24: return "quente"
    if 17 <= temp < 24: return "neutro"
    if 10 <= temp < 17: return "frio"
    return "muito_frio"

def filtrar(todas_classes, temperatura, genero_usuario):
    clima_atual = definir_tipo_clima(temperatura)
    
    recomendacao_estruturada = {
        "cabeca": [],
        "tronco": [],
        "tronco_externo": [], 
        "pernas": [],
        "pes": [],
        "corpo_inteiro": []
    }

    # Percorre o catálogo expandido
    for nome_roupa, regras in CATALOGO_REGRAS.items():
        # 1. Filtro Gênero
        if "unisex" in regras["genero"] or genero_usuario in regras["genero"]:
            
            # 2. Filtro Clima
            aceita_clima = False
            if clima_atual in regras["clima"]:
                aceita_clima = True
            # Lógica extra: Tênis sempre serve, a menos que seja muito extremo
            elif nome_roupa == "Tênis":
                aceita_clima = True
            
            if aceita_clima:
                parte = regras["parte"]
                recomendacao_estruturada[parte].append(nome_roupa)

    return recomendacao_estruturada, clima_atual