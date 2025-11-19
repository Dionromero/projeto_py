import base64
from io import BytesIO
import json
import torch
from torchvision import transforms
from PIL import Image
from dados import preparar_datasets
from clima import processar_dados_horarios


def carregar_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predizer_imagem(caminho_img, modelo_path="modelo_final.pth"):
    # Carrega datasets apenas para obter vetores de clima e número de classes
    train_ds, _, num_classes = preparar_datasets()  
    clima_vector = train_ds.clima.unsqueeze(0)      

    # Cria o modelo e carrega os pesos
    modelo = processar_dados_horarios(num_classes, len(train_ds.clima))
    modelo.load_state_dict(torch.load(modelo_path, map_location="cpu"))
    modelo.eval()

    # Transform padrão (ImageNet)
    transform = carregar_transform()

    # Abre a imagem
    img = Image.open(caminho_img).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # Faz predição
    with torch.no_grad():
        out = modelo(img_tensor, clima_vector)
        _, pred = torch.max(out, 1)

    print(f"Classe prevista: {pred.item()}")
    return pred.item()


def imagem_para_json(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    bytes_img = buffer.getvalue()

    img_b64 = base64.b64encode(bytes_img).decode("utf-8")

    return json.dumps({
        "image_base64": img_b64
    })
