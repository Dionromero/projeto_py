import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from datasets import load_dataset
from PIL import Image
import numpy as np

# %% Carrega o dataset 'clothes_simplified' do Hugging Face
dataset = load_dataset("samokosik/clothes_simplified")

# %% Acesso aos splits
df_train = dataset['train']
df_test = dataset['test']

# %% Exemplo de acesso a um item
print(df_train[0])

# %% Função para extrair features simples de imagem (opcional)
def extract_features(img_path, size=(32, 32)):
    img = Image.open(img_path).resize(size)
    return np.array(img).flatten()

# %% Criação de um Dataset customizado para PyTorch
class ClothesDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item['image']).convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# %% Criação dos DataLoaders
transform = ToTensor()
train_dataset = ClothesDataset(df_train, transform=transform)
test_dataset = ClothesDataset(df_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %% Verificação rápida
for images, labels in train_loader:
    print(images.shape, labels.shape)
    break
