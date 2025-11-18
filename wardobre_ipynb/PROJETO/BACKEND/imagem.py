import base64
from io import BytesIO
import json

def imagem_para_json(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    bytes_img = buffer.getvalue()

    img_b64 = base64.b64encode(bytes_img).decode("utf-8")

    return json.dumps({
        "image_base64": img_b64
    })
