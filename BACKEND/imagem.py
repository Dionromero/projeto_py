# imagem.py - predição (aceita caminho para arquivo de imagem)
import torch
from PIL import Image
from torchvision import transforms
import base64
import json
import io
from modelo import ModeloVisaoClimaLite
from dados import preparar_datasets

def carregar_transform():
    return transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def json_base64_para_pil(json_str):
    # recebe string JSON ou dict, retorna PIL.Image
    if isinstance(json_str, str):
        data = json.loads(json_str)
    else:
        data = json_str
    b64 = data.get("image_base64")
    bytes_img = base64.b64decode(b64)
    return Image.open(io.BytesIO(bytes_img)).convert("RGB")

def predizer_imagem(caminho_ou_pil, modelo_path="modelo_final_b.pth"):
    # se for um caminho de arquivo
    if isinstance(caminho_ou_pil, str):
        img = Image.open(caminho_ou_pil).convert("RGB")
    else:
        # espera um PIL.Image ou JSON dict
        if isinstance(caminho_ou_pil, Image.Image):
            img = caminho_ou_pil
        else:
            img = json_base64_para_pil(caminho_ou_pil)

    train_ds, _, num_classes = preparar_datasets()  # só para clima e classes
    clima_vector = train_ds.clima.unsqueeze(0)

    modelo = ModeloVisaoClimaLite(num_classes, len(train_ds.clima), freeze_backbone=True)
    modelo.load_state_dict(torch.load(modelo_path, map_location="cpu"))
    modelo.eval()

    transform = carregar_transform()
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = modelo(img_tensor, clima_vector)
        _, pred = torch.max(out, 1)

    print("Classe prevista:", pred.item())
    return pred.item()
