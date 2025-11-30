# CATALOGO BLINDADO: Apenas peças que existem no dataset clothes-with-class
CATALOGO_REGRAS = {
    # 1. CASUAL
    "Camiseta Básica":  {"parte": "cima", "genero": ["U"], "clima": ["Q", "N"], "estilo": ["Casual"]},
    "Jeans Reto":       {"parte": "baixo", "genero": ["U"], "clima": ["N", "F", "MF"], "estilo": ["Casual"]},
    "Short Jeans":      {"parte": "baixo", "genero": ["U"], "clima": ["Q"], "estilo": ["Casual"]},
    "Moletom Hoodie":   {"parte": "casaco", "genero": ["U"], "clima": ["F", "N"], "estilo": ["Casual"]},
    "Jaqueta Jeans":    {"parte": "casaco", "genero": ["U"], "clima": ["N", "F"], "estilo": ["Casual"]},
    "Vestido Casual":   {"parte": "corpo_inteiro", "genero": ["F"], "clima": ["Q", "N"], "estilo": ["Casual"]},

    # 2. FORMAL
    "Camisa Social":    {"parte": "cima", "genero": ["U"], "clima": ["N", "Q"], "estilo": ["Formal"]},
    "Calça Social":     {"parte": "baixo", "genero": ["U"], "clima": ["N", "F"], "estilo": ["Formal"]},
    "Blazer":           {"parte": "casaco", "genero": ["U"], "clima": ["N", "F"], "estilo": ["Formal"]},
    "Terno":            {"parte": "corpo_inteiro", "genero": ["M"], "clima": ["N", "F"], "estilo": ["Formal"]},
    "Vestido Social":   {"parte": "corpo_inteiro", "genero": ["F"], "clima": ["Q", "N"], "estilo": ["Formal"]},

    # 3. ESPORTIVO
    "Regata Sport":     {"parte": "cima", "genero": ["U"], "clima": ["Q"], "estilo": ["Esportivo"]},
    "Camiseta Dry":     {"parte": "cima", "genero": ["U"], "clima": ["Q", "N"], "estilo": ["Esportivo"]},
    "Legging":          {"parte": "baixo", "genero": ["F"], "clima": ["N", "F", "Q"], "estilo": ["Esportivo"]},
    "Short Esportivo":  {"parte": "baixo", "genero": ["U"], "clima": ["Q"], "estilo": ["Esportivo"]},
    "Jaqueta Sport":    {"parte": "casaco", "genero": ["U"], "clima": ["F", "N"], "estilo": ["Esportivo"]},
}

def definir_tipo_clima(temp):
    if temp >= 22: return "Q"
    if 15 <= temp < 22: return "N"
    if 10 <= temp < 15: return "F"
    return "MF"

def filtrar(todas_classes, temperatura, genero_usuario, estilo_usuario):
    clima_atual = definir_tipo_clima(temperatura)
    
    # Define permissões de gênero
    if genero_usuario == "male": generos_permitidos = ["M", "U"]
    elif genero_usuario == "female": generos_permitidos = ["F", "U"]
    else: generos_permitidos = ["M", "F", "U"]

    recomendacao = {"cima": [], "baixo": [], "casaco": [], "corpo_inteiro": []}

    for nome, regras in CATALOGO_REGRAS.items():
        if estilo_usuario not in regras["estilo"]: continue

        eh_compativel = any(g in generos_permitidos for g in regras["genero"])
        if not eh_compativel: continue

        clima_ok = False
        if clima_atual in regras["clima"]: clima_ok = True
        if clima_atual == "F" and "MF" in regras["clima"]: clima_ok = True
        if clima_atual == "N" and "Q" in regras["clima"] and regras["parte"] == "cima": clima_ok = True

        if clima_ok:
            recomendacao[regras["parte"]].append(nome)

    return recomendacao, clima_atual