import torch
import numpy as np  # noqa: F401
from flask import Blueprint, jsonify
from torchvision import transforms
from datasets import load_dataset
from modelo import ModeloVisaoClimaLite 
from clima_api import obter_dados_clima, processar_dados_para_tabela, criar_dataframes, processar_dados_horarios  # noqa: F401

recommendations_bp = Blueprint('recommendations', __name__)

# Configurações
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "modelo_final_b.pth"
LABEL_MAP = {0: "Camiseta", 1: "Calça", 2: "Vestido", 3: "Jaqueta", 4: "Saia", 5: "Short", 
             6: "Suéter", 7: "Blusa", 8: "Meia", 9: "Sapato", 10: "Chapéu", 11: "Acessório"}

modelo_global = None
exemplos_cache = {}

def inicializar_modelo():
    global modelo_global, exemplos_cache
    if modelo_global is not None: return  # noqa: E701

    try:
        print("Carregando dataset para cache...")
        # Lógica de carregar dataset (mantida do seu código original)
        dataset = load_dataset("samokosik/clothes_simplified", split="train")
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        needed = set(LABEL_MAP.keys())
        for item in dataset:
            lbl = item['label']
            if lbl in needed and lbl not in exemplos_cache:
                img_t = transform(item['image']).unsqueeze(0)
                exemplos_cache[lbl] = img_t.to(DEVICE)
            if len(exemplos_cache) == len(needed): break  # noqa: E701
        
        # Carregar Modelo (Usando a classe importada, não redefinida)
        print("Carregando modelo...")
        clima_dim = 3
        # Usa a classe que veio do `from modelo import ...`
        modelo_global = ModeloVisaoClimaLite(num_classes=12, clima_dim=clima_dim, freeze_backbone=True)
        modelo_global.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        modelo_global.to(DEVICE)
        modelo_global.eval()
        print("Sistema pronto.")
    except Exception as e:
        print(f"Erro ao inicializar: {e}")

@recommendations_bp.route('/', methods=['GET'])
def recomendar():
    if modelo_global is None: inicializar_modelo()  # noqa: E701
    
    # Usa funções importadas do clima_api.py
    dados = obter_dados_clima("Curitiba") 
    _, df_forecast = criar_dataframes(dados)
    vec = processar_dados_horarios(df_forecast)
    clima_tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    scores = {}
    with torch.no_grad():
        for label, img_tensor in exemplos_cache.items():
            out = modelo_global(img_tensor, clima_tensor)
            prob = torch.softmax(out, dim=1)[0][label].item()
            scores[label] = float(prob)

    melhor = max(scores, key=scores.get)
    return jsonify({
        "recomendacao_id": int(melhor),
        "nome_roupa": LABEL_MAP.get(melhor),
        "scores": scores
    })