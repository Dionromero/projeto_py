import os
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import numpy as np
from clima_api import obter_dados_clima, criar_dataframes, processar_dados_horarios

IMG_SIZE = (128, 128)

class ClothesDataset(Dataset):
    def __init__(self, split, clima_vector):
        # Carrega o dataset do Hugging Face
        dataset = load_dataset("samokosik/clothes_simplified")[split]
        self.data = dataset
        
        # Transformações da imagem
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        
        # Vetor de clima (igual para todas as imagens neste setup simplificado)
        self.clima = torch.tensor(clima_vector, dtype=torch.float32)

        # Mapeamento de labels para garantir índices 0, 1, 2...
        labels = [item["label"] for item in dataset]
        classes = sorted(list(set(labels)))
        self.label_map = {old: new for new, old in enumerate(classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Processa a imagem
        img = self.transform(item["image"])
        label = self.label_map[item["label"]]
        
        # 💡 GERAÇÃO DO ID: Usa o índice numérico formatado com 8 dígitos
        image_id = f"{idx:08d}"
        
        # Retorna 4 valores: Imagem, Clima, Label e o ID
        return img, self.clima, torch.tensor(label, dtype=torch.long), image_id


def preparar_datasets(local="Curitiba"):
    # Busca dados de clima atuais
    dados = obter_dados_clima(local)
    _, df_forecast = criar_dataframes(dados)
    clima_vector = processar_dados_horarios(df_forecast)

    # Cria datasets de treino e teste
    train = ClothesDataset("train", clima_vector)
    test = ClothesDataset("test", clima_vector)

    # Conta número de classes
    num_classes = len(set([item["label"] for item in load_dataset("samokosik/clothes_simplified")["train"]]))
    
    return train, test, num_classes