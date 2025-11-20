from flask import Flask, jsonify
from flask_cors import CORS
import torch
from modelo import ModeloVisaoClimaLite
from dados import preparar_datasets
from clima_api import obter_dados_clima, criar_dataframes, processar_dados_horarios, processar_dados_para_tabela

app = Flask(__name__)
CORS(app) # Permite comunicação entre Frontend e Backend

print("\n🔧 Inicializando servidor… Carregando modelo e dataset...")

# ------------------------------------------------------------
# 1) Carregar dataset e Clima
# ------------------------------------------------------------
train_ds, _, num_classes = preparar_datasets("Curitiba")
clima_dim = len(train_ds.clima)

# ------------------------------------------------------------
# Mapeamento de Nomes (Labels -> Texto Legível)
# ------------------------------------------------------------
label_to_name = {
    0: "Camiseta", 
    1: "Calça", 
    2: "Vestido", 
    3: "Jaqueta", 
    4: "Saia", 
    5: "Short", 
    6: "Suéter", 
    7: "Blusa", 
    8: "Meia", 
    9: "Sapato", 
    10: "Chapéu", 
    11: "Acessório" 
}

# ------------------------------------------------------------
# 2) Carregar modelo
# ------------------------------------------------------------
modelo = ModeloVisaoClimaLite(num_classes, clima_dim, freeze_backbone=True)
try:
    modelo.load_state_dict(torch.load("modelo_final_b.pth", map_location="cpu"))
    modelo.eval()
    print("✔ Modelo carregado com sucesso.")
except FileNotFoundError:
    print("❌ ERRO CRÍTICO: Arquivo 'modelo_final_b.pth' não encontrado.")

# ------------------------------------------------------------
# 3) Selecionar exemplos por classe
# ------------------------------------------------------------
exemplos_por_classe = {}

# Loop ajustado para receber 4 valores do dados.py
for img, clima, label, image_id in train_ds:
    label_item = label.item()
    
    if label_item not in exemplos_por_classe:
        class_folder = str(label_item)
        
        # Constrói o caminho relativo que o frontend usará para buscar a imagem
        frontend_img_path = f"images/{class_folder}/image_{image_id}.jpg"
        
        exemplos_por_classe[label_item] = {
            'img_tensor': img.unsqueeze(0), 
            'imagem_path': frontend_img_path 
        }
            
    # Para assim que tiver 1 exemplo de cada classe
    if len(exemplos_por_classe) == num_classes:
        break

print(f"✔ Exemplos carregados ({len(exemplos_por_classe)} classes).")

# ------------------------------------------------------------
# 4) Endpoint /recomendar
# ------------------------------------------------------------
@app.get("/recomendar")
def recomendar():
    # 1. Obter clima atualizado
    dados = obter_dados_clima("Curitiba")
    _, df_forecast = criar_dataframes(dados)
    clima_vec = processar_dados_horarios(df_forecast)
    clima_vec = torch.tensor(clima_vec).unsqueeze(0)

    scores = {}

    # 2. Rodar modelo para cada classe
    with torch.no_grad():
        for label, exemplo_data in exemplos_por_classe.items():
            # Pega o tensor da imagem de exemplo
            exemplo_img_tensor = exemplo_data['img_tensor']
            
            # Predição
            out = modelo(exemplo_img_tensor, clima_vec)
            
            # Calcula probabilidade
            prob = torch.softmax(out, dim=1)[0][label].item()
            scores[label] = float(prob)

    if not scores:
        return jsonify({"erro": "Não foi possível gerar recomendações"}), 500

    # 3. Escolher a melhor classe
    melhor_classe = max(scores, key=scores.get)
    
    # 4. Preparar dados para o frontend
    recomendacao_nome = label_to_name.get(melhor_classe, f"Classe {melhor_classe}")
    recomendacao_img_path = exemplos_por_classe[melhor_classe].get('imagem_path')

    return jsonify({
        "recomendacao": int(melhor_classe),
        "nome_roupa": recomendacao_nome,
        "imagem_path": recomendacao_img_path,
        "scores": scores
    })

# ------------------------------------------------------------
# 5) Endpoint /api/clima
# ------------------------------------------------------------
@app.get("/api/clima")
def clima_tabela():
    dados = obter_dados_clima("Curitiba")
    dados_tabela = processar_dados_para_tabela(dados)
    return jsonify(dados_tabela)


if __name__ == "__main__":
    print("🚀 Servidor pronto! http://localhost:5000/")
    app.run(host="0.0.0.0", port=5000)