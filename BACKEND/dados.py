import os
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset
import torch

IMG_SIZE = (128, 128)

class ClothesDataset(Dataset):
    def __init__(self, split):
        # 1. Carrega o dataset do Hugging Face
        dataset = load_dataset("samokosik/clothes_simplified")[split]
        self.data = dataset
        
        # 2. Define Transformações
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((140, 140)),
                transforms.RandomCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

        # 3. CORREÇÃO DO ERRO AQUI:
        # Antes estava: for item in split (errado, split é uma string "train")
        # Agora está: for item in dataset (certo, dataset é a lista de dados)
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
        image_id = f"{idx:08d}"
        
        # Retorna apenas Imagem, Label e ID (Sem Clima)
        return img, torch.tensor(label, dtype=torch.long), image_id


def preparar_datasets():
    # Carrega datasets sem passar vetor de clima
    train = ClothesDataset("train")
    test = ClothesDataset("test")

    # Conta número de classes para configurar o modelo
    # (Pega uma amostra do treino para contar)
    amostra = load_dataset("samokosik/clothes_simplified")["train"]
    labels = [item["label"] for item in amostra]
    num_classes = len(set(labels))
    
    return train, test, num_classes