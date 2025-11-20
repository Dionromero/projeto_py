# dados.py (forma B - leve)
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import numpy as np
from clima_api import obter_dados_clima, criar_dataframes, processar_dados_horarios

IMG_SIZE = (128, 128)

class ClothesDataset(Dataset):
    def __init__(self, split, clima_vector):
        dataset = load_dataset("samokosik/clothes_simplified")[split]
        self.data = dataset
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        # clima_vector é 1D numpy array (float32)
        self.clima = torch.tensor(clima_vector, dtype=torch.float32)

        # map labels -> 0..C-1 (consistente)
        labels = [item["label"] for item in dataset]
        classes = sorted(list(set(labels)))
        self.label_map = {old: new for new, old in enumerate(classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # item["image"] já é PIL.Image do HF dataset
        img = self.transform(item["image"])
        label = self.label_map[item["label"]]
        # clima repetido para cada amostra (mesmo vetor para todas)
        return img, self.clima, label


def preparar_datasets(local="Curitiba"):
    dados = obter_dados_clima(local)
    _, df_forecast = criar_dataframes(dados)
    clima_vector = processar_dados_horarios(df_forecast)  # numpy array 1D

    train = ClothesDataset("train", clima_vector)
    test = ClothesDataset("test", clima_vector)

    # número de classes (a partir do split train)
    num_classes = len(set([item["label"] for item in load_dataset("samokosik/clothes_simplified")["train"]]))
    return train, test, num_classes
