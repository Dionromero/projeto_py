import sys
import os
import io
import base64
import random
import torch
from flask import Blueprint, jsonify, request

current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

try:
    from clima_api import obter_dados_clima
except ImportError: pass

try:
    from datasets import load_dataset
    TEM_DATASETS = True
except ImportError:
    TEM_DATASETS = False

# Importa a IA
from brain import NeuralStylist, preparar_dados_entrada, ROUPAS_CIMA, ROUPAS_BAIXO, ROUPAS_CASACO
from train_brain import treinar_agora

recommendations_bp = Blueprint('recommendations', __name__)

DATASET_STREAM = None
IA_MODELO = None

# --- CARREGA OU TREINA A IA NA HORA ---
def iniciar_ia():
    global IA_MODELO
    caminho_modelo = "cerebro_estilista.pth"
    
    if not os.path.exists(caminho_modelo):
        treinar_agora()
        
    IA_MODELO = NeuralStylist()
    try:
        IA_MODELO.load_state_dict(torch.load(caminho_modelo, weights_only=True))
        IA_MODELO.eval() # Modo de uso
        print(" IA Pronta para uso.")
    except:
        print(" Modelo corrompido. Retreinando...")
        treinar_agora()
        IA_MODELO.load_state_dict(torch.load(caminho_modelo, weights_only=True))
        IA_MODELO.eval()

iniciar_ia()

# --- CONFIGURAÇÃO DE IMAGENS ---
if TEM_DATASETS:
    try:
        DATASET_STREAM = load_dataset("deadprogram/clothes-with-class", split="train", streaming=True)
    except: pass

# Mapeamento para Imagens (IA -> Dataset)
NOME_PARA_CATEGORIA = {
    "Camiseta Básica": "tees", "Camiseta Oversized": "tees", "Camiseta Dry": "tees", "Regata Sport": "tanks",
    "Camisa Social": "shirts", "Suéter": "sweaters", "Top Cropped": "tops", 
    "Camisa Polo": "shirts", "Camisa Jeans": "shirts", "Blusa de Seda": "tops", "Bata Estampada": "tops",
    "Camisa Xadrez": "shirts", "Top de Brilho": "tops",
    
    "Jeans Reto": "jeans", "Jeans Rasgado": "jeans", "Calça Cargo": "pants", "Calça Social": "pants",
    "Legging": "leggings", "Short Esportivo": "shorts", "Short Jeans": "shorts",
    "Saia Mídi": "skirts", "Calça Jogger": "joggers", "Calça de Couro": "pants",
    "Calça Wide Leg": "pants", "Saia Longa": "skirts", "Minissaia": "skirts", "Bermuda": "shorts",
    "Calça Chino": "pants", "Calça Grossa": "pants",
    
    "Moletom Hoodie": "hoodies", "Jaqueta Jeans": "jackets", "Blazer": "blazers", 
    "Jaqueta Sport": "jackets", "Cardigan": "cardigans", "Sobretudo": "coats", "Jaqueta Bomber": "jackets", 
    "Jaqueta de Couro": "jackets", "Jaqueta Curta": "jackets", "Jaqueta Utilitária": "jackets", 
    "Jaqueta de Pelo": "jackets", "Trench Coat": "coats", "Nada": None
}

# Backup de Imagens (Se dataset falhar)
IMAGENS_RESERVA = {
    "pes": "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=600&q=80",
    "cima": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=600&q=80",
    "baixo": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=600&q=80",
    "casaco": "https://images.unsplash.com/photo-1551028919-ac7edd05b6ea?w=600&q=80"
}

def obter_imagem_fallback(texto):
    if not texto or texto == "-" or texto == "Nada": return "https://placehold.co/1x1/ffffff/ffffff"
    return f"https://placehold.co/600x800/1e293b/ffffff?font=montserrat&text={texto.upper()}"

def imagem_para_base64(pil_img):
    try:
        data = io.BytesIO()
        if pil_img.mode != 'RGB': pil_img = pil_img.convert('RGB')
        pil_img.save(data, "JPEG")
        encoded = base64.b64encode(data.getvalue())
        return "data:image/jpeg;base64," + encoded.decode('utf-8')
    except: return None

def buscar_imagem(nome_peca, genero):
    if not nome_peca or nome_peca == "Nada": return None

    if DATASET_STREAM:
        categoria = NOME_PARA_CATEGORIA.get(nome_peca)
        if categoria:
            try:
                # Busca rápida no stream
                iterator = DATASET_STREAM.skip(random.randint(0, 300)).take(60)
                genero_alvo = "women" if genero == "female" else "men"
                if genero == "unisex": genero_alvo = None
                
                for item in iterator:
                    if item['clothing'] == categoria:
                        if not genero_alvo or item['gender'] == genero_alvo:
                            return imagem_para_base64(item['image'])
            except: pass

    return obter_imagem_fallback(nome_peca)

@recommendations_bp.route('/', methods=['GET'])
def recomendar():
    genero = request.args.get('genero', 'unisex')
    local = request.args.get('local', 'Curitiba')
    estilo = request.args.get('estilo', 'Casual')

    try:
        dados = obter_dados_clima(local)
        temp = dados['current']['temp_c'] if dados else 22.0
    except: temp = 22.0

    # --- IA DECIDINDO ---
    tensor_input = preparar_dados_entrada(temp, genero, estilo)
    
    with torch.no_grad():
        out_c, out_b, out_k = IA_MODELO(tensor_input)
        
        # IA Escolhe os índices mais prováveis
        idx_cima = torch.argmax(out_c, dim=1).item()
        idx_baixo = torch.argmax(out_b, dim=1).item()
        idx_casaco = torch.argmax(out_k, dim=1).item()
        
        peca_cima = ROUPAS_CIMA[idx_cima]
        peca_baixo = ROUPAS_BAIXO[idx_baixo]
        peca_casaco = ROUPAS_CASACO[idx_casaco]

    if peca_casaco == "Nada": peca_casaco = None

    imagens = {
        "cima": buscar_imagem(peca_cima, genero),
        "baixo": buscar_imagem(peca_baixo, genero),
        "casaco": buscar_imagem(peca_casaco, genero) if peca_casaco else None,
        "pes": IMAGENS_RESERVA["pes"] # Sapato fixo
    }

    look = {
        "cima": peca_cima,
        "baixo": peca_baixo,
        "casaco": peca_casaco,
        "pes": "Calçado Ideal"
    }

    cat_clima = "Frio" if temp < 16 else "Quente" if temp > 23 else "Ameno"

    return jsonify({
        "status": "success", 
        "temperatura": temp, 
        "categoria_clima": f"{cat_clima} (IA)", 
        "look": look, 
        "imagens": imagens
    })