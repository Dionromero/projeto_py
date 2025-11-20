from flask import Flask, jsonify
from flask_cors import CORS
import torch
from modelo import ModeloVisaoClimaLite
from dados import preparar_datasets
from clima_api import obter_dados_clima, criar_dataframes, processar_dados_horarios

app = Flask(__name__)
CORS(app)


print("\n🔧 Inicializando servidor… Carregando modelo e dataset...")

# ------------------------------------------------------------
# 1) Carregar dataset uma única vez
# ------------------------------------------------------------
train_ds, _, num_classes = preparar_datasets("Curitiba")
clima_dim = len(train_ds.clima)

# ------------------------------------------------------------
# 2) Carregar modelo somente uma vez
# ------------------------------------------------------------
modelo = ModeloVisaoClimaLite(num_classes, clima_dim, freeze_backbone=True)
modelo.load_state_dict(torch.load("modelo_final_b.pth", map_location="cpu"))
modelo.eval()

print("✔ Modelo carregado.")

# ------------------------------------------------------------
# 3) Selecionar um exemplo por classe (apenas 1x)
# ------------------------------------------------------------
exemplos_por_classe = {}

for img, clima, label in train_ds:
    if label not in exemplos_por_classe:
        exemplos_por_classe[label] = img.unsqueeze(0)
    if len(exemplos_por_classe) == num_classes:
        break

print(f"✔ Exemplos carregados ({len(exemplos_por_classe)} classes).")

# ------------------------------------------------------------
# 4) Endpoint otimizado (rápido)
# ------------------------------------------------------------
@app.get("/recomendar")
def recomendar():

    # pegar clima real
    dados = obter_dados_clima("Curitiba")
    _, df_forecast = criar_dataframes(dados)
    clima_vec = processar_dados_horarios(df_forecast)
    clima_vec = torch.tensor(clima_vec).unsqueeze(0)

    scores = {}

    # avaliar cada classe
    with torch.no_grad():
        for label, exemplo_img in exemplos_por_classe.items():
            out = modelo(exemplo_img, clima_vec)
            prob = torch.softmax(out, dim=1)[0][label].item()
            scores[label] = float(prob)

    melhor_classe = max(scores, key=scores.get)

    return jsonify({
        "recomendacao": int(melhor_classe),
        "scores": scores
    })


if __name__ == "__main__":
    print("🚀 Servidor pronto! http://localhost:5000/recomendar")
    app.run(host="0.0.0.0", port=5000)
